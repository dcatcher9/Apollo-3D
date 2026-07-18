#!/usr/bin/env python3
"""Falsify the experimental stereo-window metric on authenticated real source frames.

The validator reuses the shared authenticated clip provenance checks and deterministic
frame selection, but renders only synthetic *exact inverse maps*.  Real image content therefore
drives the metric's contrast/frequency/orientation weighting while the desired border geometry is
known exactly.  It covers central crossed disparity (benign), graded crossed cuts, sign separation,
source-derived contrast/orientation/frequency stimuli, aspect-fit bars, raw-U clamps, folds, and
forward-coverage exclusion.

Passing this suite is necessary, never sufficient.  The Disney-inspired weighting is not a
display-calibrated psychophysical model, so this command never changes thresholds or label status
and always reports ``training_label_qualification = blocked``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from PIL import __version__ as PILLOW_VERSION


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbs_stereo_window_metrics as stereo_window  # noqa: E402
import sbsbench  # noqa: E402
import authenticated_metric_sources as authenticated_sources  # noqa: E402


SCHEMA = 1
DISPARITY_LEVELS_PCT = (-0.5, -1.0, -2.0, -3.0)
CONTRAST_LEVELS = (0.15, 0.35, 0.65, 1.0)


def mapping_shape(eye_width, eye_height, source_width, source_height,
                  scale_x=1.0, scale_y=1.0):
    return {
        "schema": 1,
        "dtype": "float32-le",
        "layout": "row-major",
        "channels": ["raw_reproject_source_u_normalized"],
        "width": 2 * int(eye_width),
        "height": int(eye_height),
        "eye_width": int(eye_width),
        "eye_height": int(eye_height),
        "source_width": int(source_width),
        "source_height": int(source_height),
        "content_scale_x": float(scale_x),
        "content_scale_y": float(scale_y),
    }


def identity_eye(shape):
    width, height = shape["eye_width"], shape["eye_height"]
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    lo_x = 0.5 * (1.0 - shape["content_scale_x"])
    source_u = (output_u - lo_x) / shape["content_scale_x"]
    return np.broadcast_to(source_u, (height, width)).copy()


def signed_maps(shape, disparity_pct, *, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    """Return packed raw inverse maps with actual binocular disparity xR-xL."""
    base = identity_eye(shape)
    left, right = base.copy(), base.copy()
    height, width = base.shape
    region = np.zeros(base.shape, dtype=bool)
    region[int(round(y0 * height)):int(round(y1 * height)),
           int(round(x0 * width)):int(round(x1 * width))] = True
    left[region] += float(disparity_pct) / 200.0
    right[region] -= float(disparity_pct) / 200.0
    return np.concatenate((left, right), axis=1)


def _measure(mapping, shape, source, **kwargs):
    return stereo_window.measure_stereo_window_violation(
        mapping, shape, source, **kwargs)


def _check(name, status, detail, **values):
    if status not in ("pass", "fail", "abstain"):
        raise ValueError(status)
    return {"name": name, "status": status, "detail": detail, "values": values}


def _zero_check(name, metrics, keys, tolerance=1e-9):
    values = {key: metrics.get(key) for key in keys}
    if any(value is None for value in values.values()):
        return _check(name, "abstain", "metric abstained", **values)
    passed = all(abs(float(value)) <= tolerance for value in values.values())
    return _check(
        name, "pass" if passed else "fail",
        "no unsupported border risk" if passed else "benign geometry produced border risk",
        **values)


def _monotonic_check(name, levels, values, *, minimum_response=0.0, tolerance=1e-8):
    record = {"levels": [float(value) for value in levels],
              "responses": [None if value is None else float(value) for value in values]}
    if any(value is None or not np.isfinite(value) for value in values):
        return _check(name, "abstain", "one or more levels lacked qualified support", **record)
    if float(values[-1]) <= minimum_response:
        return _check(name, "abstain", "real source lacks enough visible evidence", **record)
    monotonic = all(float(after) + tolerance >= float(before)
                    for before, after in zip(values[:-1], values[1:]))
    responsive = float(values[-1]) > float(values[0]) + minimum_response
    passed = monotonic and responsive
    return _check(
        name, "pass" if passed else "fail",
        "response is ordered and nontrivial" if passed else "response is not ordered",
        **record)


def _source_profile(source):
    """Select the real scanline carrying the strongest robust horizontal structure."""
    luma = sbsbench.rgb_luma(np.asarray(source, dtype=np.float32))
    centered = luma - np.median(luma, axis=1, keepdims=True)
    spread = np.percentile(centered, 95.0, axis=1) - np.percentile(centered, 5.0, axis=1)
    row = int(np.argmax(spread))
    profile = centered[row].astype(np.float32)
    scale = float(np.percentile(np.abs(profile), 95.0))
    return None if scale < 1e-4 else profile / scale


def _normalize_profile(profile, amplitude=0.35):
    profile = np.asarray(profile, dtype=np.float32)
    profile -= float(np.mean(profile))
    scale = float(np.percentile(np.abs(profile), 95.0))
    if scale < 1e-7:
        return None
    return np.clip(profile / scale * amplitude, -amplitude, amplitude)


def _profile_image(profile, width, height, orientation):
    source_x = np.linspace(0.0, 1.0, profile.size, endpoint=True)
    if orientation == "vertical":
        target = np.linspace(0.0, 1.0, width, endpoint=True)
        signal = np.interp(target, source_x, profile).astype(np.float32)
        image = np.broadcast_to(signal[None, :], (height, width))
    elif orientation == "horizontal":
        target = np.linspace(0.0, 1.0, height, endpoint=True)
        signal = np.interp(target, source_x, profile).astype(np.float32)
        image = np.broadcast_to(signal[:, None], (height, width))
    else:
        raise ValueError(orientation)
    image = np.clip(0.5 + image, 0.0, 1.0)
    return np.repeat(image[..., None], 3, axis=2).astype(np.float32)


def _periodic_profile_image(profile, width, height, orientation):
    """Tile one real-source profile at equal pixel frequency in either orientation.

    A once-per-picture vertical profile can be locally flat exactly at the lateral cut bands,
    while its horizontal transpose exposes the full profile along those bands.  Tiling the same
    authenticated-source signal removes that phase/support confound; only orientation changes.
    """
    period = max(8, int(round(width / 32.0)))
    source_x = np.linspace(0.0, 1.0, profile.size, endpoint=False)
    target_x = np.linspace(0.0, 1.0, period, endpoint=False)
    period_signal = np.interp(target_x, source_x, profile, period=1.0).astype(np.float32)
    output_length = width if orientation == "vertical" else height
    edge_extent = max(4, int(round(0.04 * width)))
    best = None
    for offset in range(period):
        signal = np.resize(np.roll(period_signal, offset), output_length)
        if orientation == "vertical":
            score = float(np.var(signal[:edge_extent]) + np.var(signal[-edge_extent:]))
        else:
            score = float(np.var(signal))
        if best is None or score > best[0]:
            best = (score, signal)
    signal = best[1]
    if orientation == "vertical":
        image = np.broadcast_to(signal[None, :], (height, width))
    elif orientation == "horizontal":
        image = np.broadcast_to(signal[:, None], (height, width))
    else:
        raise ValueError(orientation)
    image = np.clip(0.5 + image, 0.0, 1.0)
    return np.repeat(image[..., None], 3, axis=2).astype(np.float32)


def _frequency_profile(profile, lower_cycle, upper_cycle):
    profile = np.asarray(profile, dtype=np.float32)
    spectrum = np.fft.rfft(profile - np.mean(profile))
    cycles = np.arange(spectrum.size)
    keep = (cycles >= lower_cycle) & (cycles <= upper_cycle)
    filtered = np.fft.irfft(np.where(keep, spectrum, 0.0), n=profile.size).astype(np.float32)
    return _normalize_profile(filtered)


def _controlled_source_checks(source, shape):
    checks = []
    crossed_map = signed_maps(shape, -2.0)
    prefix = "experimental_stereo_window_"

    # Preserve real content while scaling its physical contrast about neutral gray.  Per-frame
    # normalization in the metric would fail this check by making all four responses equal.
    contrast_metrics = []
    for level in CONTRAST_LEVELS:
        adjusted = np.clip(0.5 + float(level) * (source - 0.5), 0.0, 1.0)
        contrast_metrics.append(_measure(crossed_map, shape, adjusted))
    checks.append(_monotonic_check(
        "graded_real_source_contrast",
        CONTRAST_LEVELS,
        [metrics[prefix + "crossed_burden_pct"] for metrics in contrast_metrics],
        minimum_response=0.002))

    profile = _source_profile(source)
    if profile is None:
        checks.append(_check(
            "source_derived_orientation", "abstain",
            "authenticated frame has no usable one-dimensional structure"))
        checks.append(_check(
            "source_derived_frequency", "abstain",
            "authenticated frame has no usable one-dimensional structure"))
        return checks

    normalized = _normalize_profile(profile)
    vertical_source = _periodic_profile_image(
        normalized, shape["source_width"], shape["source_height"], "vertical")
    horizontal_source = _periodic_profile_image(
        normalized, shape["source_width"], shape["source_height"], "horizontal")
    vertical = _measure(crossed_map, shape, vertical_source)[prefix + "crossed_burden_pct"]
    horizontal = _measure(crossed_map, shape, horizontal_source)[prefix + "crossed_burden_pct"]
    if vertical is None or horizontal is None or max(vertical, horizontal) <= 0.002:
        checks.append(_check(
            "source_derived_orientation", "abstain",
            "derived orientation pair has insufficient visible burden",
            vertical=vertical, horizontal=horizontal))
    else:
        # The local cut band and nonlinear contrast saturation keep the aggregate ratio below the
        # per-pixel 1/0.70 orientation factor on some real profiles.  Direction plus a 5% margin
        # still falsifies an ignored or reversed orientation channel.
        passed = vertical >= horizontal * 1.05
        checks.append(_check(
            "source_derived_orientation", "pass" if passed else "fail",
            "horizontal structures are downweighted" if passed
            else "orientation response contradicts the documented weighting",
            vertical=vertical, horizontal=horizontal))

    bands = {
        "low": (1, 4),
        "mid": (24, min(64, max(24, profile.size // 4))),
        "high": (96, min(160, max(96, profile.size // 2 - 1))),
    }
    responses = {}
    for name, (lower, upper) in bands.items():
        band = _frequency_profile(profile, lower, upper) if upper >= lower else None
        if band is None:
            responses[name] = None
            continue
        image = _profile_image(
            band, shape["source_width"], shape["source_height"], "vertical")
        responses[name] = _measure(
            crossed_map, shape, image)[prefix + "crossed_burden_pct"]
    if any(value is None for value in responses.values()) or max(responses.values()) <= 0.002:
        checks.append(_check(
            "source_derived_frequency", "abstain",
            "one or more authenticated-source bands had insufficient energy", **responses))
    else:
        passed = (responses["mid"] > responses["low"] * 1.10
                  and responses["mid"] > responses["high"] * 1.02)
        checks.append(_check(
            "source_derived_frequency", "pass" if passed else "fail",
            "picture-space response has the expected mid-band preference" if passed
            else "spatial-frequency response is not ordered", **responses))
    return checks


def evaluate_sample(clip, frame_path, max_width):
    source = authenticated_sources.load_frame(frame_path, max_width)
    source_height, source_width = source.shape[:2]
    shape = mapping_shape(
        source_width, source_height, source_width, source_height)
    prefix = "experimental_stereo_window_"
    checks = []

    identity = signed_maps(shape, 0.0)
    identity_metrics = _measure(identity, shape, source)
    checks.append(_zero_check(
        "identity_benign", identity_metrics,
        (prefix + "crossed_area_pct", prefix + "uncrossed_area_pct")))

    central_metrics = _measure(
        signed_maps(shape, -2.0, x0=0.30, x1=0.70), shape, source)
    checks.append(_zero_check(
        "central_crossed_disparity_benign", central_metrics,
        (prefix + "crossed_burden_pct", prefix + "crossed_area_pct",
         prefix + "crossed_largest_component_pct")))

    graded = [_measure(signed_maps(shape, level), shape, source)
              for level in DISPARITY_LEVELS_PCT]
    checks.append(_monotonic_check(
        "graded_crossed_geometry_area",
        [abs(value) for value in DISPARITY_LEVELS_PCT],
        [metrics[prefix + "crossed_area_pct"] for metrics in graded],
        minimum_response=0.05))
    checks.append(_monotonic_check(
        "graded_crossed_perceptual_burden",
        [abs(value) for value in DISPARITY_LEVELS_PCT],
        [metrics[prefix + "crossed_burden_pct"] for metrics in graded],
        minimum_response=0.002))

    uncrossed = _measure(signed_maps(shape, 2.0), shape, source)
    uncrossed_values = {
        "crossed_area": uncrossed[prefix + "crossed_area_pct"],
        "uncrossed_area": uncrossed[prefix + "uncrossed_area_pct"],
    }
    if any(value is None for value in uncrossed_values.values()):
        status = "abstain"
    else:
        status = ("pass" if abs(uncrossed_values["crossed_area"]) <= 1e-9
                  and uncrossed_values["uncrossed_area"] > 0.05 else "fail")
    checks.append(_check(
        "crossed_uncrossed_sign_separation", status,
        "opposite sign remains a separate diagnostic", **uncrossed_values))

    checks.extend(_controlled_source_checks(source, shape))

    # The same source/effect under an eye-height letterbox must not let black bars vote.
    boxed_height = max(source_height + 2, int(round(source_height / 0.75)))
    scale_y = source_height / float(boxed_height)
    boxed_shape = mapping_shape(
        source_width, boxed_height, source_width, source_height, scale_y=scale_y)
    full = graded[2]
    # Keep the exact camera geometry/raw shift fixed.  The black rows must not create additional
    # cut area.  The perceptual burden is expected to fall because the shared production contract
    # normalizes the same horizontal pixel disparity by the taller full-eye aspect ratio.
    boxed = _measure(signed_maps(boxed_shape, -2.0), boxed_shape, source)
    area_key = prefix + "crossed_area_pct"
    burden_key = prefix + "crossed_burden_pct"
    comparable = (area_key, burden_key)
    bar_values = {"full_" + key: full[key] for key in comparable}
    bar_values.update({"boxed_" + key: boxed[key] for key in comparable})
    if any(value is None for value in bar_values.values()):
        bar_status = "abstain"
    else:
        area_stable = abs(float(full[area_key]) - float(boxed[area_key])) <= 0.03
        burden_ratio = float(boxed[burden_key]) / max(float(full[burden_key]), 1e-9)
        # The 0.75 height ratio is not exact after the metric's 0.1% dead zone, so retain a
        # deliberately narrow but non-formula-circular interval around the expected response.
        burden_scaled = 0.65 <= burden_ratio <= 0.85
        bar_status = "pass" if area_stable and burden_scaled else "fail"
        bar_values["boxed_to_full_burden_ratio"] = burden_ratio
    checks.append(_check(
        "letterbox_exclusion", bar_status,
        "bars preserve cut geometry while full-eye normalization scales burden"
        if bar_status == "pass"
        else "letterbox result changed or abstained", **bar_values))

    # Raw clamps and a folded run are invalid inverse evidence.  They may reduce support, but an
    # identity field with either corruption must never manufacture signed border risk.
    clamped = identity.copy()
    edge = max(2, int(round(0.05 * source_width)))
    for eye_index in range(2):
        offset = eye_index * source_width
        clamped[:, offset:offset + edge] = -0.2
        clamped[:, offset + source_width - edge:offset + source_width] = 1.2
    clamped_metrics = _measure(clamped, shape, source)
    clamp_check = _zero_check(
        "raw_u_clamp_exclusion", clamped_metrics,
        (prefix + "crossed_area_pct", prefix + "uncrossed_area_pct"))
    clamp_check["values"]["support_pct"] = clamped_metrics[prefix + "support_pct"]
    if clamp_check["status"] == "pass" and not (
            clamped_metrics[prefix + "support_pct"]
            < identity_metrics[prefix + "support_pct"] - 1.0):
        clamp_check["status"] = "fail"
        clamp_check["detail"] = "raw clamps did not reduce exact support"
    checks.append(clamp_check)

    folded = identity.copy()
    fold_width = max(3, source_width // 4)
    folded[:, :fold_width] = folded[:, :fold_width][:, ::-1]
    folded_metrics = _measure(folded, shape, source)
    fold_check = _zero_check(
        "fold_exclusion", folded_metrics,
        (prefix + "crossed_area_pct", prefix + "uncrossed_area_pct"))
    fold_check["values"]["support_pct"] = folded_metrics[prefix + "support_pct"]
    if fold_check["status"] == "pass" and not (
            folded_metrics[prefix + "support_pct"]
            < identity_metrics[prefix + "support_pct"] - 5.0):
        fold_check["status"] = "fail"
        fold_check["detail"] = "fold did not reduce exact support"
    checks.append(fold_check)

    crossed_map = signed_maps(shape, -2.0)
    coverage = np.ones(crossed_map.shape, dtype=bool)
    coverage_edge = max(2, int(round(0.12 * source_width)))
    for eye_index in range(2):
        offset = eye_index * source_width
        coverage[:, offset:offset + coverage_edge] = False
        coverage[:, offset + source_width - coverage_edge:offset + source_width] = False
    covered = _measure(crossed_map, shape, source, coverage_mask=coverage)
    coverage_check = _zero_check(
        "forward_coverage_exclusion", covered,
        (prefix + "crossed_area_pct", prefix + "crossed_burden_pct"))
    coverage_check["values"]["support_pct"] = covered[prefix + "support_pct"]
    checks.append(coverage_check)

    # Simulated FP16 HDR validates that nonlinear preview weighting remains finite and geometry
    # independent.  It is explicitly not an HDR perceptual-threshold qualification.
    linear_hdr = (sbsbench._srgb_to_linear(source) * 6.0).astype(np.float16).astype(np.float32)
    hdr = _measure(
        crossed_map, shape, linear_hdr,
        source_sample_transform=sbsbench._hdr_preview_rgb)
    hdr_values = {
        "sdr_area": full[prefix + "crossed_area_pct"],
        "hdr_area": hdr[prefix + "crossed_area_pct"],
        "hdr_burden": hdr[prefix + "crossed_burden_pct"],
    }
    if any(value is None or not np.isfinite(value) for value in hdr_values.values()):
        hdr_status = "abstain"
    else:
        hdr_status = ("pass" if abs(hdr_values["sdr_area"] - hdr_values["hdr_area"]) <= 1e-9
                      else "fail")
    checks.append(_check(
        "fractional_hdr_post_sample_transform", hdr_status,
        "HDR transform preserves exact geometry and returns finite perceptual evidence",
        **hdr_values))

    return {
        "clip": clip["id"],
        "frame": os.path.basename(frame_path),
        "source_sha256": authenticated_sources._sample_sha256(frame_path),
        "analysis_source_geometry": [source_width, source_height],
        "checks": checks,
    }


def _evaluate_job(arguments):
    return evaluate_sample(*arguments)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(clip_roots=None, frames_per_clip=1, max_width=512, max_clips=None,
                 workers=None):
    if max_width is not None and max_width < 64:
        raise ValueError("max_width must be at least 64")
    if frames_per_clip < 1:
        raise ValueError("frames_per_clip must be positive")
    roots = clip_roots or [authenticated_sources.DEFAULT_CLIPS_ROOT]
    clips = authenticated_sources.discover_clips(roots)
    if max_clips is not None:
        if max_clips < 1:
            raise ValueError("max_clips must be positive")
        clips = clips[:max_clips]
    jobs = []
    clip_manifest = {}
    for clip in clips:
        selected = authenticated_sources.deterministic_frame_sample(
            clip["frames"], frames_per_clip)
        clip_manifest[clip["id"]] = {
            "name": clip["meta"]["name"],
            "dataset": clip["meta"].get("dataset", "committed core fixture"),
            "citation": clip["meta"].get("citation"),
            "license_note": clip["meta"].get("license_note"),
            "suite": clip["meta"].get("suite", "core-repository"),
            "selected_frames": [os.path.basename(path) for path in selected],
        }
        jobs.extend((clip, frame_path, max_width) for frame_path in selected)

    requested_workers = min(4, os.cpu_count() or 1) if workers is None else int(workers)
    if requested_workers < 1:
        raise ValueError("workers must be positive")
    worker_count = min(requested_workers, len(jobs))
    if worker_count > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            samples = list(executor.map(_evaluate_job, jobs))
    else:
        samples = [_evaluate_job(job) for job in jobs]

    counts = Counter(check["status"] for sample in samples for check in sample["checks"])
    checks_by_name = {}
    for name in sorted({check["name"] for sample in samples for check in sample["checks"]}):
        statuses = Counter(
            check["status"] for sample in samples for check in sample["checks"]
            if check["name"] == name)
        checks_by_name[name] = {
            "samples": sum(statuses.values()),
            "passed": statuses["pass"],
            "failed": statuses["fail"],
            "abstained": statuses["abstain"],
        }

    return {
        "schema": SCHEMA,
        "purpose": "authenticated-real-source falsification of stereo-window metrics",
        "training_label_qualification": "blocked",
        "auto_promotes_labels": False,
        "qualification_note": (
            "Passing controlled exact-map checks is necessary but not sufficient. Independent "
            "real renderer failures and display/viewer psychophysics remain required."),
        "limitations": [
            "contrast/orientation/frequency stimuli are deterministic transforms of real frames",
            "picture-space frequency is not calibrated cycles per degree",
            "simulated HDR validates evaluator order, not HDR perceptual thresholds",
        ],
        "configuration": {
            "clip_roots": [os.path.abspath(root) for root in roots],
            "frames_per_clip": int(frames_per_clip),
            "max_width": int(max_width),
            "max_clips": max_clips,
            "workers": worker_count,
            "disparity_levels_pct": list(DISPARITY_LEVELS_PCT),
            "contrast_levels": list(CONTRAST_LEVELS),
        },
        "implementation": {
            "metric_sha256": _file_sha256(stereo_window.__file__),
            "validator_sha256": _file_sha256(__file__),
            "source_authenticator_sha256": _file_sha256(authenticated_sources.__file__),
            "numpy_version": np.__version__,
            "pillow_version": PILLOW_VERSION,
        },
        "summary": {
            "clips": len(clips),
            "samples": len(samples),
            "checks": sum(counts.values()),
            "passed": counts["pass"],
            "failed": counts["fail"],
            "abstained": counts["abstain"],
            "status": "pass" if counts["fail"] == 0 else "fail",
        },
        "checks_by_name": checks_by_name,
        "clip_manifest": clip_manifest,
        "samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("core", "extended", "both"), default="core",
        help="authenticated suite selection when --clips-root is omitted")
    parser.add_argument("--clips-root", action="append", dest="clip_roots")
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output", help="write deterministic JSON report")
    parser.add_argument(
        "--strict", action="store_true",
        help="return failure when any check fails or abstains")
    args = parser.parse_args()
    roots = args.clip_roots or authenticated_sources.suite_roots(args.suite)
    report = build_report(
        roots, frames_per_clip=args.frames_per_clip, max_width=args.max_width,
        max_clips=args.max_clips, workers=args.workers)
    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    failed = report["summary"]["failed"]
    abstained = report["summary"]["abstained"]
    return 1 if failed or (args.strict and abstained) else 0


if __name__ == "__main__":
    raise SystemExit(main())
