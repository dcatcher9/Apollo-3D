#!/usr/bin/env python3
"""Optional exact-map-registered NVIDIA FLIP appearance oracle for Apollo SBS output.

FLIP is useful here only after removing intended stereo geometry.  Each rendered eye is paired
with a clean eye regenerated from the mono source through that eye's exact production source-U
map.  Actual and clean eyes are then inverted onto one common source raster.  Only source samples
that are uniquely visible in *both* eyes may vote, so disparity, bars, clamps, folds, and holes do
not masquerade as appearance defects.

The official ``flip-evaluator`` package is imported lazily.  Missing or incompatible packages,
insufficient support, and HDR preview PNGs produce explicit unavailable/abstained payloads.  This
module deliberately runs LDR-FLIP only: Apollo's current HDR PNG artifact is a display preview of
linear scRGB, not the lossless HDR buffer FLIP-HDR requires.

No global mean is computed or reported.  The diagnostic pools localized worst-eye p99 and error
area, plus an interocular error-map imbalance.  Every result is experimental and stamped
``training_label_eligible=false``; it is not part of evaluator policy or model labels.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

try:
    from . import sbs_interocular_phase_chroma as _exact
except ImportError:  # Direct execution from tools/sbsbench.
    import sbs_interocular_phase_chroma as _exact


SCHEMA = 1
ORACLE = "nvidia-flip-exact-appearance"
ROLE = "optional_eval_only_experimental_diagnostic"
DEFAULT_PPD = 67.0
DEFAULT_AREA_THRESHOLD = 0.05
DEFAULT_MIN_SUPPORT_PIXELS = 1024
DEFAULT_MIN_SUPPORT_PCT = 5.0


class FlipUnavailable(RuntimeError):
    """The optional official FLIP implementation cannot be used."""


def _base_payload(status: str, reason: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "oracle": ORACLE,
        "status": status,
        "role": ROLE,
        "qualification": "experimental_diagnostic_only",
        "training_label_eligible": False,
    }
    if reason:
        payload["reason"] = reason
    return payload


def load_official_flip() -> tuple[Callable[..., Any], str]:
    """Load the official NVIDIA package lazily and return its evaluate function/version."""
    try:
        import flip_evaluator
    except (ImportError, OSError) as error:
        raise FlipUnavailable(
            "official optional dependency 'flip-evaluator' is unavailable") from error
    evaluate = getattr(flip_evaluator, "evaluate", None)
    if not callable(evaluate):
        raise FlipUnavailable("flip-evaluator does not expose evaluate()")
    try:
        version = importlib.metadata.version("flip-evaluator")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return evaluate, version


def _flip_error_map(evaluate: Callable[..., Any], reference: np.ndarray,
                    test: np.ndarray, ppd: float) -> np.ndarray:
    """Call FLIP without its magma map or forbidden global-mean reduction."""
    result = evaluate(
        np.ascontiguousarray(reference, dtype=np.float32),
        np.ascontiguousarray(test, dtype=np.float32),
        "LDR",
        inputsRGB=True,
        applyMagma=False,
        computeMeanError=False,
        parameters={"ppd": float(ppd)},
    )
    if not isinstance(result, tuple) or not result:
        raise RuntimeError("flip-evaluator returned an invalid result")
    error = np.asarray(result[0], dtype=np.float32)
    if error.ndim == 3 and error.shape[2] == 1:
        error = error[..., 0]
    if error.shape != reference.shape[:2] or not np.isfinite(error).all():
        raise RuntimeError(
            f"flip-evaluator returned invalid error map {error.shape}")
    return np.clip(error, 0.0, 1.0)


def _observe_filter_radius(evaluate: Callable[..., Any], ppd: float) -> int:
    """Empirically bound the installed FLIP implementation's LDR spatial footprint.

    FLIP's filters vary with pixels-per-degree and implementation version.  Rather than copying
    a private kernel constant, probe achromatic and opponent-colour impulses, find the farthest
    nonzero response, and let callers add a conservative two-pixel safety margin.
    """
    if not np.isfinite(ppd) or ppd <= 0.0 or ppd > 300.0:
        raise ValueError("ppd must be finite and in (0, 300]")
    size = max(129, int(np.ceil(ppd * 3.5)))
    if size % 2 == 0:
        size += 1
    center = size // 2
    reference = np.full((size, size, 3), 0.18, dtype=np.float32)
    probes = (
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    observed = 0
    for colour in probes:
        test = reference.copy()
        test[center, center] = colour
        error = _flip_error_map(evaluate, reference, test, ppd)
        threshold = max(1e-7, float(error.max()) * 1e-7)
        rows, columns = np.where(error > threshold)
        if not rows.size:
            raise RuntimeError("FLIP impulse probe produced no response")
        if (rows.min() == 0 or rows.max() == size - 1
                or columns.min() == 0 or columns.max() == size - 1):
            raise RuntimeError("FLIP impulse response reached probe boundary")
        observed = max(
            observed,
            int(np.max(np.abs(rows - center))),
            int(np.max(np.abs(columns - center))),
        )
    return observed


@lru_cache(maxsize=16)
def _cached_official_filter_support(ppd: float, version: str) -> tuple[int, int]:
    del version  # Version remains part of the cache key.
    evaluate, _ = load_official_flip()
    observed = _observe_filter_radius(evaluate, ppd)
    return observed, observed + 2


def measure_filter_support(ppd: float = DEFAULT_PPD,
                           dependency: tuple[Callable[..., Any], str] | None = None
                           ) -> tuple[int, int]:
    """Return ``(observed_impulse_radius, conservative_erosion_radius)``."""
    if dependency is None:
        _, version = load_official_flip()
        return _cached_official_filter_support(float(ppd), version)
    evaluate, _ = dependency
    observed = _observe_filter_radius(evaluate, float(ppd))
    return observed, observed + 2


def _as_ldr_rgb(image: np.ndarray, name: str) -> np.ndarray:
    value = _exact._as_rgb(image, name)
    minimum = float(np.min(value))
    maximum = float(np.max(value))
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(
            f"{name} is outside LDR sRGB [0,1]: min={minimum}, max={maximum}")
    return np.clip(value, 0.0, 1.0)


def _hdr_preview_reason(stats: dict[str, Any] | None) -> str | None:
    if not stats:
        return None
    format_name = str(stats.get("format", "")).strip().lower()
    source_kind = str(stats.get("hdr_source_kind", "")).strip().lower()
    explicit_preview = any(
        stats.get(key) is True
        for key in ("preview", "preview_only", "png_is_preview", "hdr_preview")
    )
    if explicit_preview:
        return "hdr_output_stats marks the PNG as an HDR preview"
    if "scrgb" in format_name or "fp16" in format_name or "hdr" in format_name:
        return (
            f"hdr_output_stats format {format_name!r} requires lossless linear HDR evidence; "
            "the current PNG is only a preview")
    if source_kind and source_kind not in ("sdr", "native-sdr", "sdr-srgb-8bit"):
        return (
            f"hdr_output_stats source kind {source_kind!r} is HDR; preview PNGs are not "
            "valid FLIP-HDR inputs")
    return None


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask.copy()
    size = 2 * radius + 1
    padded = np.pad(mask.astype(np.int32), radius, mode="constant")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant")
    integral = integral.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64)
    summed = (integral[size:, size:] - integral[:-size, size:]
              - integral[size:, :-size] + integral[:-size, :-size])
    return summed == size * size


def _register_eye(actual: np.ndarray, expected: np.ndarray, mapping: np.ndarray,
                  shape: dict[str, Any], width: int, height: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combined = np.concatenate((actual, expected), axis=2)
    registered, unique = _exact._register_rgb(
        combined, mapping, shape, width, height)
    return registered[..., :3], registered[..., 3:], unique


def _empty_metrics(area_threshold: float) -> dict[str, Any]:
    suffix = f"{int(round(area_threshold * 1000)):03d}"
    return {
        "flip_worst_eye_p99": None,
        f"flip_worst_eye_area_gt_{suffix}_pct": None,
        "flip_interocular_error_imbalance_p99": None,
        f"flip_interocular_area_imbalance_gt_{suffix}_pct": None,
    }


def measure_flip_appearance(
        source_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray,
        map_left: np.ndarray, map_right: np.ndarray, shape: dict[str, Any], *,
        hdr_output_stats: dict[str, Any] | None = None, ppd: float = DEFAULT_PPD,
        area_threshold: float = DEFAULT_AREA_THRESHOLD,
        min_support_pixels: int = DEFAULT_MIN_SUPPORT_PIXELS,
        min_support_pct: float = DEFAULT_MIN_SUPPORT_PCT,
        dependency: tuple[Callable[..., Any], str] | None = None,
        filter_radius_override: int | None = None, return_maps: bool = False,
        ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure exact-map-registered LDR appearance faults with official NVIDIA FLIP.

    ``shape`` and both source-U maps must describe the exact production warp.  The common source
    raster uses the source image's native geometry; no analysis resize changes FLIP's PPD.
    """
    if not 0.0 < area_threshold < 1.0:
        raise ValueError("area_threshold must be in (0,1)")
    if min_support_pixels < 1 or not 0.0 <= min_support_pct <= 100.0:
        raise ValueError("invalid support floor")
    preview_reason = _hdr_preview_reason(hdr_output_stats)
    if preview_reason:
        payload = _base_payload("abstained", preview_reason)
        payload.update({
            "dynamic_range": "HDR-preview-abstained",
            "metrics": _empty_metrics(area_threshold),
        })
        return (payload, {}) if return_maps else payload

    try:
        evaluate, version = dependency or load_official_flip()
    except FlipUnavailable as error:
        payload = _base_payload("unavailable", str(error))
        payload.update({"dynamic_range": "LDR", "metrics": _empty_metrics(area_threshold)})
        return (payload, {}) if return_maps else payload

    source = _as_ldr_rgb(source_rgb, "source_rgb")
    left = _as_ldr_rgb(left_rgb, "left_rgb")
    right = _as_ldr_rgb(right_rgb, "right_rgb")
    if left.shape != right.shape:
        raise ValueError(f"eye geometry differs: {left.shape} != {right.shape}")
    map_left = np.asarray(map_left, dtype=np.float32)
    map_right = np.asarray(map_right, dtype=np.float32)
    if map_left.shape != left.shape[:2] or map_right.shape != right.shape[:2]:
        raise ValueError("each exact source-U map must match its rendered eye")
    if not np.isfinite(map_left).all() or not np.isfinite(map_right).all():
        raise ValueError("exact source-U maps contain non-finite values")

    expected_left = _exact._sample_source_eye(source, map_left, shape)
    expected_right = _exact._sample_source_eye(source, map_right, shape)
    height, width = source.shape[:2]
    actual_left, reference_left, unique_left = _register_eye(
        left, expected_left, map_left, shape, width, height)
    actual_right, reference_right, unique_right = _register_eye(
        right, expected_right, map_right, shape, width, height)
    mutual = unique_left & unique_right

    if filter_radius_override is None:
        if dependency is None:
            observed_radius, erosion_radius = _cached_official_filter_support(
                float(ppd), version)
        else:
            observed_radius, erosion_radius = measure_filter_support(
                ppd, (evaluate, version))
    else:
        if filter_radius_override < 0:
            raise ValueError("filter_radius_override must be nonnegative")
        erosion_radius = int(filter_radius_override)
        observed_radius = max(0, erosion_radius - 2)
    support = _erode(mutual, erosion_radius)
    support_count = int(np.count_nonzero(support))
    support_pct = float(support_count / max(support.size, 1) * 100.0)

    dependency_info = {
        "package": "flip-evaluator",
        "version": version,
        "implementation": "official NVIDIA FLIP Python API",
    }
    support_info = {
        "unique_mutual_count_before_erosion": int(np.count_nonzero(mutual)),
        "count": support_count,
        "pct": support_pct,
        "observed_impulse_radius_px": observed_radius,
        "erosion_radius_px": erosion_radius,
        "erosion_method": "installed-FLIP impulse footprint plus two-pixel margin",
    }
    if support_count < min_support_pixels or support_pct < min_support_pct:
        payload = _base_payload(
            "abstained",
            "insufficient uniquely mutual support after FLIP filter-footprint erosion",
        )
        payload.update({
            "dynamic_range": "LDR-sRGB",
            "pixels_per_degree": float(ppd),
            "dependency": dependency_info,
            "support": support_info,
            "metrics": _empty_metrics(area_threshold),
        })
        maps = {"unique_mutual": mutual, "pooled_support": support}
        return (payload, maps) if return_maps else payload

    # Neutralize all invalid pixels *before* FLIP; the measured erosion then removes any spatial
    # filter response that could still cross that neutralization boundary.
    left_test = np.where(mutual[..., None], actual_left, reference_left)
    right_test = np.where(mutual[..., None], actual_right, reference_right)
    left_reference = np.where(mutual[..., None], reference_left, 0.0)
    right_reference = np.where(mutual[..., None], reference_right, 0.0)
    left_test = np.where(mutual[..., None], left_test, left_reference)
    right_test = np.where(mutual[..., None], right_test, right_reference)

    left_error = _flip_error_map(evaluate, left_reference, left_test, ppd)
    right_error = _flip_error_map(evaluate, right_reference, right_test, ppd)
    left_values = left_error[support]
    right_values = right_error[support]
    left_p99 = float(np.quantile(left_values, 0.99))
    right_p99 = float(np.quantile(right_values, 0.99))
    left_area = float(np.mean(left_values >= area_threshold) * 100.0)
    right_area = float(np.mean(right_values >= area_threshold) * 100.0)
    interocular = np.abs(left_error - right_error)[support]
    suffix = f"{int(round(area_threshold * 1000)):03d}"
    metrics = {
        "flip_worst_eye_p99": max(left_p99, right_p99),
        f"flip_worst_eye_area_gt_{suffix}_pct": max(left_area, right_area),
        "flip_interocular_error_imbalance_p99": float(np.quantile(interocular, 0.99)),
        f"flip_interocular_area_imbalance_gt_{suffix}_pct": abs(left_area - right_area),
    }
    payload = _base_payload("ok")
    payload.update({
        "dynamic_range": "LDR-sRGB",
        "pixels_per_degree": float(ppd),
        "area_threshold": float(area_threshold),
        "dependency": dependency_info,
        "support": support_info,
        "per_eye_diagnostics": {
            "left_p99": left_p99,
            "right_p99": right_p99,
            f"left_area_gt_{suffix}_pct": left_area,
            f"right_area_gt_{suffix}_pct": right_area,
        },
        "metrics": metrics,
    })
    maps = {
        "reference_left": reference_left,
        "reference_right": reference_right,
        "actual_left": actual_left,
        "actual_right": actual_right,
        "unique_mutual": mutual,
        "pooled_support": support,
        "flip_left": left_error,
        "flip_right": right_error,
        "flip_interocular_imbalance": np.abs(left_error - right_error),
    }
    return (payload, maps) if return_maps else payload


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sbs", type=Path, required=True)
    parser.add_argument("--warp-map", type=Path, required=True)
    parser.add_argument("--warp-shape", type=Path, required=True)
    parser.add_argument("--hdr-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence-npz", type=Path)
    parser.add_argument("--ppd", type=float, default=DEFAULT_PPD)
    args = parser.parse_args()

    with Image.open(args.source) as image:
        source = np.asarray(image.convert("RGB"))
    with Image.open(args.sbs) as image:
        sbs = np.asarray(image.convert("RGB"))
    if sbs.shape[1] % 2:
        raise ValueError(f"SBS width must be even, got {sbs.shape[1]}")
    eye_width = sbs.shape[1] // 2
    left, right = sbs[:, :eye_width], sbs[:, eye_width:]
    shape = _load_json(args.warp_shape)
    if not isinstance(shape, dict):
        raise ValueError("warp-shape must contain a JSON object")
    packed = np.fromfile(args.warp_map, dtype=np.float32)
    expected_count = sbs.shape[0] * sbs.shape[1]
    if packed.size != expected_count:
        raise ValueError(
            f"warp map contains {packed.size} floats, expected {expected_count}")
    packed = packed.reshape(sbs.shape[:2])
    result = measure_flip_appearance(
        source, left, right, packed[:, :eye_width], packed[:, eye_width:], shape,
        hdr_output_stats=_load_json(args.hdr_stats), ppd=args.ppd,
        return_maps=args.evidence_npz is not None,
    )
    if args.evidence_npz is not None:
        payload, maps = result
        args.evidence_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.evidence_npz, **maps)
    else:
        payload = result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["status"] in ("ok", "abstained") else 2


if __name__ == "__main__":
    raise SystemExit(_main())
