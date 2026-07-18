#!/usr/bin/env python3
"""Validate the experimental interocular phase metric on real source frames.

This standalone corruption validator does not edit evaluator thresholds or qualify training
labels.  It renders clean intended disparity from real source frames, applies controlled one-eye
faults and benign controls, and fails closed when an expected response or abstention is absent.
The JSON report retains every scenario metric so later human/headset calibration can audit the
detector rather than trusting a single pass/fail bit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_interocular_phase_chroma as phase_metric  # noqa: E402
import authenticated_metric_sources as authenticated_sources  # noqa: E402


SCHEMA = 3
PHASE = "interocular_phase_orientation_burden_pct"
PHASE_OK = "interocular_phase_orientation_evidence_sufficient"
PREFERRED_CLIPS = (
    "aigen_cogvideox_rain",
    "anime_morevna_closeup",
    "c647",
    "c841",
)
FOOTPRINT_FRACTIONS = (0.0008, 0.005, 0.01, 0.02, 0.05)


def _select_clips(clips, max_clips):
    if max_clips < 1:
        raise ValueError("max_clips must be positive")
    by_id = {clip["id"]: clip for clip in clips}
    selected = [by_id[name] for name in PREFERRED_CLIPS if name in by_id]
    selected_ids = {clip["id"] for clip in selected}
    selected.extend(
        clip for clip in clips
        if clip["id"] not in selected_ids and clip["meta"].get("expected_flat") is not True)
    return selected[:max_clips]


def intended_maps(width, height, fractional_px=0.43):
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    disparity = 0.020 * (0.40 + 0.60 * np.sin(np.pi * v) ** 2)
    left = np.broadcast_to(u[None, :], (height, width)).copy()
    right = left.copy()
    left += 0.5 * disparity[:, None] + fractional_px / width
    right -= 0.5 * disparity[:, None] + fractional_px / width
    return left.astype(np.float32), right.astype(np.float32)


def _measure(source, left, right, maps, shape, *, transform=None, return_maps=False,
             analysis_width=640):
    return phase_metric.measure_interocular_phase_chroma(
        source, left, right, maps[0], maps[1], shape,
        source_sample_transform=transform,
        max_analysis_width=analysis_width,
        max_analysis_height=int(round(analysis_width * 9.0 / 16.0)),
        min_phase_pixels=64, return_maps=return_maps)


def _render(source, maps, shape, transform=None):
    eyes = []
    for mapping in maps:
        eye = phase_metric._sample_source_eye(source, mapping, shape)
        eyes.append(transform(eye) if transform is not None else eye)
    return eyes


def _blur_rgb(image, radius):
    return np.stack([
        phase_metric._registration._box_mean(image[..., channel], radius)
        for channel in range(3)
    ], axis=2)


def _detail_patch(source):
    height, width = source.shape[:2]
    luma = source @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
    gx, gy = phase_metric._central_gradients(luma)
    energy = gx * gx + gy * gy
    patch_width = max(24, int(round(width * 0.42)))
    patch_height = max(16, int(round(height * 0.46)))
    half_width, half_height = patch_width // 2, patch_height // 2
    allowed = energy.copy()
    allowed[:half_height] = -1.0
    allowed[-half_height:] = -1.0
    allowed[:, :half_width] = -1.0
    allowed[:, -half_width:] = -1.0
    center_y, center_x = np.unravel_index(np.argmax(allowed), allowed.shape)
    y0 = int(np.clip(center_y - half_height, 0, height - patch_height))
    x0 = int(np.clip(center_x - half_width, 0, width - patch_width))
    return slice(y0, y0 + patch_height), slice(x0, x0 + patch_width)


def _localized_patch(source, footprint_fraction, center=None):
    """Return a thin, high-detail patch with the requested picture-area footprint."""
    if not 0.0 < footprint_fraction < 1.0:
        raise ValueError("footprint_fraction must be in (0, 1)")
    height, width = source.shape[:2]
    area = max(4, int(round(height * width * footprint_fraction)))
    patch_width = max(2, int(round(np.sqrt(area / 4.0))))
    patch_height = max(2, int(np.ceil(area / patch_width)))
    patch_width = min(patch_width, max(2, width - 2))
    patch_height = min(patch_height, max(2, height - 2))

    half_width, half_height = patch_width // 2, patch_height // 2
    if center is None:
        luma = source @ np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
        gx, gy = phase_metric._central_gradients(luma)
        energy = gx * gx + gy * gy
        allowed = energy.copy()
        allowed[:half_height + 1] = -1.0
        allowed[-half_height - 1:] = -1.0
        allowed[:, :half_width + 1] = -1.0
        allowed[:, -half_width - 1:] = -1.0
        center_y, center_x = np.unravel_index(np.argmax(allowed), allowed.shape)
    else:
        center_y, center_x = center
    y0 = int(np.clip(center_y - half_height, 0, height - patch_height))
    x0 = int(np.clip(center_x - half_width, 0, width - patch_width))
    return slice(y0, y0 + patch_height), slice(x0, x0 + patch_width)


def _localized_precision(evidence, key, patch_y, patch_x, source_shape, floor):
    """Fraction of above-floor detector response inside a conservative injected-patch halo."""
    conflict = np.nan_to_num(evidence[key], nan=0.0)
    active = conflict > floor
    if not np.any(active):
        return 0.0, 0
    analysis_height, analysis_width = active.shape
    source_height, source_width = source_shape[:2]
    scale_x = analysis_width / float(source_width)
    scale_y = analysis_height / float(source_height)
    # Registration/filter support expands a real fault by a few analysis pixels.  This halo is
    # proportional to the normalized analysis raster, not the source resolution.
    margin = max(4, int(round(analysis_width * 0.02)))
    left = max(0, int(np.floor(patch_x.start * scale_x)) - margin)
    right = min(analysis_width, int(np.ceil(patch_x.stop * scale_x)) + margin)
    top = max(0, int(np.floor(patch_y.start * scale_y)) - margin)
    bottom = min(analysis_height, int(np.ceil(patch_y.stop * scale_y)) + margin)
    expected = np.zeros(active.shape, dtype=bool)
    expected[top:bottom, left:right] = True
    active_count = int(np.count_nonzero(active))
    return float(np.count_nonzero(active & expected) / active_count), active_count


def _ladder_check(name, fractions, records, metric, baseline, minimum_response):
    values = [record.get(metric) for record in records]
    sufficient = all(value is not None and np.isfinite(value) for value in values)
    # A tiny numerical/local-content wobble is allowed, but a footprint increase must not erase
    # a material amount of accumulated coherent burden.
    tolerance = 0.05
    monotonic = sufficient and all(
        after + tolerance >= before for before, after in zip(values, values[1:]))
    responsive = (sufficient and baseline.get(metric) is not None
                  and values[-1] >= baseline[metric] + minimum_response)
    return _check(
        name, bool(monotonic and responsive), fractions=list(fractions), values=values,
        baseline=baseline.get(metric), monotonic=bool(monotonic), responsive=bool(responsive))


def _check(name, passed, **evidence):
    return {"name": name, "status": "pass" if passed else "fail", **evidence}


def _below(metrics, key, limit):
    return metrics.get(key) is not None and metrics[key] < limit


def _responds(metrics, baseline, key, delta):
    return (metrics.get(key) is not None and baseline.get(key) is not None
            and metrics[key] >= baseline[key] + delta)


def validate_source(source, source_id, analysis_width=640):
    height, width = source.shape[:2]
    shape = {"content_scale_x": 1.0, "content_scale_y": 1.0}
    maps = intended_maps(width, height)
    clean_eyes = _render(source, maps, shape)
    clean = _measure(
        source, *clean_eyes, maps, shape, analysis_width=analysis_width)
    patch_y, patch_x = _detail_patch(source)
    shift = max(2, int(round(width * 0.015)))

    phase_eye = clean_eyes[1].copy()
    phase_patch = phase_eye[patch_y, patch_x].copy()
    phase_eye[patch_y, patch_x] = np.roll(phase_patch, shift, axis=1)
    localized_phase = _measure(
        source, clean_eyes[0], phase_eye, maps, shape, analysis_width=analysis_width)

    orientation_eye = clean_eyes[1].copy()
    orientation_eye[patch_y, patch_x] = np.flip(
        orientation_eye[patch_y, patch_x], axis=0)
    localized_orientation = _measure(
        source, clean_eyes[0], orientation_eye, maps, shape,
        analysis_width=analysis_width)

    phase_footprints = []
    footprint_patches = []
    phase_localization = (0.0, 0)
    anchor_y, anchor_x = _localized_patch(source, max(FOOTPRINT_FRACTIONS))
    anchor = ((anchor_y.start + anchor_y.stop) // 2,
              (anchor_x.start + anchor_x.stop) // 2)
    local_shift = max(1, int(round(width * 0.006)))
    shifted_eye = np.roll(clean_eyes[1], local_shift, axis=1)
    for footprint in FOOTPRINT_FRACTIONS:
        local_y, local_x = _localized_patch(source, footprint, center=anchor)
        footprint_patches.append({
            "fraction": footprint,
            "x": [local_x.start, local_x.stop],
            "y": [local_y.start, local_y.stop],
        })

        phase_fault = clean_eyes[1].copy()
        phase_fault[local_y, local_x] = shifted_eye[local_y, local_x]
        phase_result = _measure(
            source, clean_eyes[0], phase_fault, maps, shape,
            return_maps=footprint == 0.01, analysis_width=analysis_width)
        if footprint == 0.01:
            phase_result, phase_evidence = phase_result
            phase_localization = _localized_precision(
                phase_evidence, "phase_orientation_conflict_pct",
                local_y, local_x, source.shape, 5.0)
        phase_footprints.append(phase_result)

    exposure_left = clean_eyes[0] * np.asarray((0.76, 0.69, 0.73), dtype=np.float32)
    exposure_right = clean_eyes[1] * np.asarray((1.12, 1.25, 1.17), dtype=np.float32)
    exposure_wb = _measure(
        source, exposure_left, exposure_right, maps, shape,
        analysis_width=analysis_width)

    blur_trials = []
    for fraction in (0.0125, 0.025, 0.05, 0.10):
        radius = max(2, int(round(width * fraction)))
        blurred = _blur_rgb(clean_eyes[1], radius)
        blur_metrics = _measure(
            source, clean_eyes[0], blurred, maps, shape,
            analysis_width=analysis_width)
        blur_trials.append({"radius": radius, "metrics": blur_metrics})
        if blur_metrics[PHASE_OK] == 0.0:
            break
    blur_detail = blur_trials[-1]

    bar_shape = {"content_scale_x": 0.75, "content_scale_y": 1.0}
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    bar_u = (output_u - 0.125) / 0.75
    bar_map = np.broadcast_to(bar_u[None, :], (height, width)).copy()
    bar_maps = (bar_map, bar_map.copy())
    bar_eyes = _render(source, bar_maps, bar_shape)
    content = (output_u >= 0.125) & (output_u <= 0.875)
    hostile = ~np.broadcast_to(content[None, :], (height, width))
    yy, xx = np.indices((height, width))
    pattern = ((xx + 3 * yy) % 7 < 3).astype(np.float32)
    bar_eyes[0][hostile] = np.stack(
        (pattern[hostile], 1.0 - pattern[hostile], pattern[hostile]), axis=1)
    bar_eyes[1][hostile] = np.stack(
        (1.0 - pattern[hostile], pattern[hostile], 1.0 - pattern[hostile]), axis=1)
    bars = _measure(
        source, *bar_eyes, bar_maps, bar_shape, analysis_width=analysis_width)

    linear = np.where(
        source <= 0.04045,
        source / 12.92,
        ((source + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32) * 6.0

    def tonemap(value):
        return value / (1.0 + value)

    hdr_eyes = _render(linear, maps, shape, tonemap)
    hdr = _measure(
        linear, *hdr_eyes, maps, shape, transform=tonemap,
        analysis_width=analysis_width)

    scenarios = {
        "clean_intended_disparity": clean,
        "localized_phase": localized_phase,
        "localized_orientation": localized_orientation,
        "global_exposure_white_balance": exposure_wb,
        "blur_detail_imbalance": blur_detail,
        "hostile_bars": bars,
        "hdr_post_sample_transform": hdr,
        "phase_footprint_ladder": phase_footprints,
    }
    checks = [
        _check("clean_evidence", clean[PHASE_OK] == 100.0, metrics=clean),
        _check("clean_low_conflict", _below(clean, PHASE, 2.0), metrics=clean),
        _check("localized_phase_response",
               localized_phase[PHASE_OK] == 100.0
               and _responds(localized_phase, clean, PHASE, 2.0),
               clean=clean[PHASE], corrupted=localized_phase[PHASE]),
        _check("localized_orientation_response",
               localized_orientation[PHASE_OK] == 100.0
               and _responds(localized_orientation, clean, PHASE, 2.0),
               clean=clean[PHASE], corrupted=localized_orientation[PHASE]),
        _check("global_exposure_wb_benign",
               exposure_wb[PHASE_OK] == 100.0
               and exposure_wb[PHASE] is not None and clean[PHASE] is not None
               and exposure_wb[PHASE] <= clean[PHASE] + 2.0,
               clean_phase=clean[PHASE], control_phase=exposure_wb[PHASE]),
        _check("blur_detail_abstains",
               blur_detail["metrics"][PHASE_OK] == 0.0
               and blur_detail["metrics"][PHASE] is None,
               radius=blur_detail["radius"], metrics=blur_detail["metrics"]),
        _check("bars_excluded", bars[PHASE_OK] == 100.0
               and _below(bars, PHASE, 2.0), metrics=bars),
        _check("hdr_transform_order", hdr[PHASE_OK] == 100.0
               and _below(hdr, PHASE, 1.0), metrics=hdr),
        _ladder_check(
            "phase_footprint_ladder", FOOTPRINT_FRACTIONS,
            phase_footprints, PHASE, clean, minimum_response=0.5),
        _check(
            "phase_one_percent_localization",
            phase_localization[1] > 0 and phase_localization[0] >= 0.80,
            precision=phase_localization[0], active_pixels=phase_localization[1]),
    ]
    return {
        "source": source_id,
        "geometry": {"width": width, "height": height},
        "analysis_geometry": {
            "max_width": analysis_width,
            "max_height": int(round(analysis_width * 9.0 / 16.0)),
        },
        "patch": {"x": [patch_x.start, patch_x.stop], "y": [patch_y.start, patch_y.stop]},
        "footprint_patches": footprint_patches,
        "scenarios": scenarios,
        "checks": checks,
        "passed": all(check["status"] == "pass" for check in checks),
    }


def build_report(roots, frames_per_clip=1, max_width=854, max_clips=4,
                 analysis_width=640):
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be positive")
    if max_width < 64:
        raise ValueError("max_width must be at least 64")
    if analysis_width not in (640, 960):
        raise ValueError("analysis_width must be 640 or 960")
    clips = _select_clips(authenticated_sources.discover_clips(roots), max_clips)
    samples = []
    clip_manifest = {}
    started = time.perf_counter()
    for clip in clips:
        paths = authenticated_sources.deterministic_frame_sample(
            clip["frames"], frames_per_clip)
        clip_manifest[clip["id"]] = {
            "name": clip["meta"]["name"],
            "suite": clip["meta"].get("suite", "core-repository"),
            "dataset": clip["meta"].get("dataset", "committed core fixture"),
            "citation": clip["meta"].get("citation"),
            "license_note": clip["meta"].get("license_note"),
            "selected_frames": [os.path.basename(path) for path in paths],
            "source_sha256": [authenticated_sources.sample_sha256(path) for path in paths],
        }
        for path in paths:
            source = authenticated_sources.load_frame(path, max_width)
            samples.append(validate_source(source, path, analysis_width=analysis_width))
    elapsed = time.perf_counter() - started
    checks = [check for sample in samples for check in sample["checks"]]
    failed = sum(check["status"] != "pass" for check in checks)
    return {
        "schema": SCHEMA,
        "experimental": True,
        "training_label_qualification": "blocked",
        "auto_promotes_thresholds": False,
        "configuration": {
            "clip_roots": [os.path.abspath(root) for root in roots],
            "frames_per_clip": frames_per_clip,
            "max_width": max_width,
            "max_clips": max_clips,
            "analysis_width": analysis_width,
            "analysis_height": int(round(analysis_width * 9.0 / 16.0)),
        },
        "implementation": {
            "metric_sha256": authenticated_sources.sample_sha256(phase_metric.__file__),
            "validator_sha256": authenticated_sources.sample_sha256(__file__),
            "source_authenticator_sha256": authenticated_sources.sample_sha256(
                authenticated_sources.__file__),
            "numpy_version": np.__version__,
        },
        "summary": {
            "clips": len(clips),
            "sources": len(samples),
            "checks": len(checks),
            "passed": len(checks) - failed,
            "failed": failed,
            "overall_pass": failed == 0,
            "elapsed_seconds": elapsed,
            "seconds_per_source": elapsed / max(len(samples), 1),
        },
        "clip_manifest": clip_manifest,
        "samples": samples,
    }


def build_resolution_comparison(roots, frames_per_clip=1, max_width=854, max_clips=4):
    """Run identical authenticated corruptions at 640x360 and 960x540."""
    reports = {
        width: build_report(
            roots, frames_per_clip=frames_per_clip, max_width=max_width,
            max_clips=max_clips, analysis_width=width)
        for width in (640, 960)
    }
    low_samples = {sample["source"]: sample for sample in reports[640]["samples"]}
    high_samples = {sample["source"]: sample for sample in reports[960]["samples"]}
    if set(low_samples) != set(high_samples):
        raise ValueError("analysis resolutions selected different authenticated sources")

    response_axes = (
        ("localized_phase", PHASE),
        ("localized_orientation", PHASE),
    )
    comparisons = []
    response_relative_deltas = []
    statuses_equal = True
    for source_id in sorted(low_samples):
        low, high = low_samples[source_id], high_samples[source_id]
        low_status = [check["status"] for check in low["checks"]]
        high_status = [check["status"] for check in high["checks"]]
        sample_status_equal = low_status == high_status
        statuses_equal &= sample_status_equal
        responses = {}
        for scenario, metric in response_axes:
            low_value = low["scenarios"][scenario][metric]
            high_value = high["scenarios"][scenario][metric]
            if low_value is None or high_value is None:
                relative = None
            else:
                relative = abs(low_value - high_value) / max(
                    abs(low_value), abs(high_value), 1e-6)
                response_relative_deltas.append(relative)
            responses[scenario] = {
                "metric": metric,
                "640": low_value,
                "960": high_value,
                "relative_delta": relative,
            }
        for scenario, metric in (("phase_footprint_ladder", PHASE),):
            low_ladder = low["scenarios"][scenario]
            high_ladder = high["scenarios"][scenario]
            for index, footprint in enumerate(FOOTPRINT_FRACTIONS):
                low_value = low_ladder[index][metric]
                high_value = high_ladder[index][metric]
                if low_value is None or high_value is None:
                    relative = None
                else:
                    # A fixed absolute denominator prevents clean/sub-threshold numerical noise
                    # from dominating the resolution check at the 0.08% footprint.
                    relative = abs(low_value - high_value) / max(
                        abs(low_value), abs(high_value), 0.5)
                    response_relative_deltas.append(relative)
                key = f"{scenario}_{footprint:.4f}"
                responses[key] = {
                    "metric": metric,
                    "640": low_value,
                    "960": high_value,
                    "relative_delta": relative,
                }
        comparisons.append({
            "source": source_id,
            "check_statuses_equal": sample_status_equal,
            "responses": responses,
        })

    max_response_delta = max(response_relative_deltas, default=np.inf)
    overall_pass = (
        reports[640]["summary"]["overall_pass"]
        and reports[960]["summary"]["overall_pass"]
        and statuses_equal and max_response_delta <= 0.30)
    return {
        "schema": SCHEMA,
        "purpose": "640x360 versus 960x540 sensitivity equivalence",
        "training_label_qualification": "blocked",
        "auto_promotes_thresholds": False,
        "summary": {
            "overall_pass": overall_pass,
            "check_statuses_equal": statuses_equal,
            "max_response_relative_delta": max_response_delta,
            "response_relative_delta_limit": 0.30,
            "elapsed_seconds_640": reports[640]["summary"]["elapsed_seconds"],
            "elapsed_seconds_960": reports[960]["summary"]["elapsed_seconds"],
            "speedup_640_vs_960": (
                reports[960]["summary"]["elapsed_seconds"]
                / max(reports[640]["summary"]["elapsed_seconds"], 1e-9)),
        },
        "comparisons": comparisons,
        "reports": {str(width): report for width, report in reports.items()},
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("core", "extended", "both"), default="core")
    parser.add_argument("--clips-root", action="append", dest="clip_roots")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=854)
    parser.add_argument("--max-clips", type=int, default=4)
    parser.add_argument("--analysis-width", type=int, choices=(640, 960), default=640)
    parser.add_argument(
        "--compare-960", action="store_true",
        help="run paired 640x360/960x540 sensitivity and timing validation")
    parser.add_argument("--output", help="optional JSON output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    roots = args.clip_roots or authenticated_sources.suite_roots(args.suite)
    if args.compare_960:
        report = build_resolution_comparison(
            roots, frames_per_clip=args.frames_per_clip, max_width=args.max_width,
            max_clips=args.max_clips)
    else:
        report = build_report(
            roots, frames_per_clip=args.frames_per_clip, max_width=args.max_width,
            max_clips=args.max_clips, analysis_width=args.analysis_width)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
    else:
        print(payload)
    return 0 if report["summary"]["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
