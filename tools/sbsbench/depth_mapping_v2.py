#!/usr/bin/env python3
"""NumPy comparison oracle for the production depth-coordinate-v2 GPU mapping.

The camera is acquired from the first usable DAV2 field.  Scale is a fixed authenticated
model/shape calibration and never adapts to a frame's distribution.  Acquisition may place the
zero plane at a strongly separated upper histogram valley; ambiguous fields keep the
arithmetic-mean center.  At startup or after a confirmed cut, the first usable field establishes
the center immediately; later valid, invalid, and fast-motion frames hold it until the next cut.
There is no pending state or late correction. It is exactly the zero
plane: curve-space convergence remains zero for both accepted and fallback selections.  The near
curve remains fixed, so framing and content occupancy cannot select different transfer functions.
The hard representation container is pointwise and stateless; requested gain never becomes
mutable shot state and a raw outlier cannot shrink unrelated geometry.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    from .depth_coordinate_v2_contract import (
        CALIBRATED_DEFAULTS as V2_DEFAULTS,
        MODEL_CALIBRATIONS as V2_MODEL_CALIBRATIONS,
    )
except ImportError:  # Direct script/module loading from tools/sbsbench.
    from depth_coordinate_v2_contract import (  # type: ignore
        CALIBRATED_DEFAULTS as V2_DEFAULTS,
        MODEL_CALIBRATIONS as V2_MODEL_CALIBRATIONS,
    )


DIRECT_PARALLAX_SOURCE_U_LIMIT = V2_DEFAULTS.direct_container_limit

# Scene-acquisition stage-boundary trial.  The live implementation reuses its existing 256-bin
# raw-depth histogram and merges adjacent bins before the small one-thread Otsu resolver.  These
# are deliberately implementation constants rather than user controls: uncertain evidence must
# fall back to the arithmetic mean and zero convergence.
SCENE_HISTOGRAM_SOURCE_BINS = 256
SCENE_HISTOGRAM_BINS = 128
SCENE_VALLEY_RATIO_MAX = V2_DEFAULTS.stage_valley_ratio_max


def vertical_share_coefficients(majorant_share: float) -> Tuple[float, float]:
    """Return the exact float32 coefficients consumed by the HLSL vertical blend."""

    majorant_f32 = np.float32(majorant_share)
    minorant_f32 = np.float32(np.float32(1.0) - majorant_f32)
    return float(majorant_f32), float(minorant_f32)


@dataclass(frozen=True)
class MappingV2Config:
    raw_coordinate_scale: float = V2_MODEL_CALIBRATIONS[0].raw_coordinate_scale
    collapse_abs_epsilon: float = V2_DEFAULTS.collapse_abs_epsilon
    far_tau: float = V2_DEFAULTS.far_tau
    near_log_tau: float = V2_DEFAULTS.near_log_tau
    pop_strength: float = V2_DEFAULTS.reference_pop_strength
    gain_per_pop: float = V2_DEFAULTS.gain_per_pop
    max_horizontal_slope: float = V2_DEFAULTS.max_horizontal_slope
    max_vertical_shear: float = V2_DEFAULTS.max_vertical_shear
    vertical_majorant_share: float = V2_DEFAULTS.vertical_majorant_share
    direct_container_limit: float = V2_DEFAULTS.direct_container_limit

    @property
    def parallax_gain(self) -> float:
        return self.pop_strength * self.gain_per_pop


@dataclass(frozen=True)
class CoordinateCalibration:
    center: float
    observed_std: float
    scale: float
    raw_min: float
    raw_max: float
    collapse_threshold: float
    collapsed: bool


@dataclass(frozen=True)
class SceneCoordinateSelection:
    """Deterministic diagnostics for one scene-camera acquisition attempt.

    ``selected_center`` is the selector's controlling output; ``convergence_curve`` remains an
    explicit zero-valued wire field so the selected center is the zero plane.  Everything else
    explains why the conservative selector adopted the upper valley or fell back.
    ``candidate_center`` is the stricter historical ``T - scale`` acceptance guard; an accepted
    ``selected_center`` is ``T`` itself.
    Split indices use the inclusive three-class convention: ``[0..i]``, ``[i+1..j]``, and
    ``[j+1..127]``.
    """

    observed_mean: float
    selected_center: float
    convergence_curve: float
    adopted: bool
    reason: str
    raw_min: float
    raw_max: float
    histogram_total: int
    lower_split_bin: Optional[int]
    upper_split_bin: Optional[int]
    upper_split_raw: Optional[float]
    candidate_center: Optional[float]
    otsu_score: Optional[float]
    class_counts: Optional[Tuple[int, int, int]]
    valley_numerator: Optional[float]
    middle_peak: Optional[float]
    near_peak: Optional[float]
    valley_ratio: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MappingV2Diagnostics:
    shape: Tuple[int, int]
    center_mean: float
    selected_center: float
    scene_coordinate: SceneCoordinateSelection
    observed_std: float
    raw_coordinate_scale: float
    collapse_threshold: float
    collapsed: bool
    canonical_min: float
    canonical_p01: float
    canonical_p50: float
    canonical_p99: float
    canonical_max: float
    convergence_curve: float
    curve_far_limit: float
    curve_near_limit: Optional[float]
    requested_gain: float
    effective_gain: float
    container_source_u_limit: float
    container_scale: float
    max_horizontal_slope: float
    max_step_per_texel: float
    max_vertical_shear: float
    max_vertical_step_per_texel: float
    max_adjacent_step_before_container: float
    max_adjacent_step_after_container: float
    max_adjacent_step_after_limiter: float
    max_adjacent_vertical_step_before_limiter: float
    max_adjacent_vertical_step_after_vertical_limiter: float
    max_adjacent_vertical_step_after_horizontal_limiter: float
    estimated_collar_texels_before_container: int
    estimated_collar_texels_after_container: int
    requested_min: float
    requested_max: float
    output_min: float
    output_max: float
    vertical_conditioned_raised_fraction: float
    vertical_conditioned_max_raise: float
    vertical_conditioned_lowered_fraction: float
    vertical_conditioned_max_lower: float
    horizontal_limiter_raised_fraction: float
    horizontal_limiter_max_raise: float
    horizontal_limiter_illegal_lower_count: int
    conditioner_raised_fraction: float
    conditioner_max_raise: float
    conditioner_lowered_fraction: float
    conditioner_max_lower: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MappingV2Result:
    canonical: np.ndarray
    desired_parallax: np.ndarray
    pre_limiter_parallax: np.ndarray
    post_vertical_parallax: np.ndarray
    parallax: np.ndarray
    diagnostics: MappingV2Diagnostics


def _finite_positive(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _validate_config(config: MappingV2Config) -> None:
    if not isinstance(config, MappingV2Config):
        raise TypeError("config must be a MappingV2Config")
    _finite_positive("raw_coordinate_scale", config.raw_coordinate_scale)
    _finite_positive("collapse_abs_epsilon", config.collapse_abs_epsilon)
    _finite_positive("far_tau", config.far_tau)
    _finite_positive("near_log_tau", config.near_log_tau)
    _finite_positive("pop_strength", config.pop_strength)
    _finite_positive("gain_per_pop", config.gain_per_pop)
    _finite_positive("max_horizontal_slope", config.max_horizontal_slope)
    if config.max_horizontal_slope >= 1.0:
        raise ValueError("max_horizontal_slope must be below 1 for a no-fold forward map")
    _finite_positive("max_vertical_shear", config.max_vertical_shear)
    if not math.isfinite(config.vertical_majorant_share):
        raise ValueError("vertical_majorant_share must lie strictly between zero and one")
    majorant_share, minorant_share = vertical_share_coefficients(
        config.vertical_majorant_share)
    if majorant_share <= 0.0 or minorant_share <= 0.0:
        raise ValueError(
            "vertical_majorant_share and its complement must remain positive in float32")
    _finite_positive("direct_container_limit", config.direct_container_limit)


def _require_raw_depth(raw_depth: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(raw_depth):
        raise ValueError("raw_depth must be real-valued")
    try:
        raw = np.asarray(raw_depth, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("raw_depth must be a finite, non-empty 2D numeric array") from exc
    if raw.ndim != 2 or raw.size == 0:
        raise ValueError(f"raw_depth must be a finite, non-empty 2D array, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw_depth must contain only finite values")
    return raw


def calibrate_coordinate(
        raw_depth: np.ndarray,
        config: MappingV2Config = MappingV2Config()) -> CoordinateCalibration:
    _validate_config(config)
    raw = _require_raw_depth(raw_depth)
    center = float(np.mean(raw, dtype=np.float64))
    observed_std = float(np.std(raw, dtype=np.float64))
    scale = config.raw_coordinate_scale
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    if not all(math.isfinite(value) for value in (
            center, observed_std, scale, raw_min, raw_max)):
        raise ValueError("moment calibration produced a non-finite coordinate")
    return CoordinateCalibration(
        center=center,
        observed_std=observed_std,
        scale=scale,
        raw_min=raw_min,
        raw_max=raw_max,
        collapse_threshold=config.collapse_abs_epsilon,
        collapsed=observed_std <= config.collapse_abs_epsilon,
    )


def _scene_histogram_128(raw: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Mirror ``depth_hist_cs`` and then merge its adjacent 256-bin pairs.

    Production input is FP32.  Explicit float32 operations here preserve the GPU's edge-bin
    behavior, including forcing an exact maximum into source bin 255.  ``depth_hist_cs`` accepts
    only non-negative finite samples, so a caller must compare the returned total with the raw
    texel count before treating it as complete evidence.
    """

    raw32 = np.asarray(raw, dtype=np.float32)
    raw_min32 = np.float32(np.min(raw32))
    raw_max32 = np.float32(np.max(raw32))
    raw_range32 = np.float32(raw_max32 - raw_min32)
    denominator = np.maximum(raw_range32, np.float32(1.0e-12))
    inverse_range = np.float32(
        np.float32(SCENE_HISTOGRAM_SOURCE_BINS) / denominator)
    valid = raw32 >= np.float32(0.0)
    source_counts = np.zeros(SCENE_HISTOGRAM_SOURCE_BINS, dtype=np.uint64)
    if np.any(valid):
        deltas = np.subtract(raw32[valid], raw_min32, dtype=np.float32)
        positions = np.multiply(deltas, inverse_range, dtype=np.float32)
        # HLSL's cast from a non-negative float truncates toward zero.  Clamp after conversion so
        # the exact maximum, whose scaled coordinate is normally 256, lands in the last bin.
        indices = np.minimum(
            positions.astype(np.uint32),
            np.uint32(SCENE_HISTOGRAM_SOURCE_BINS - 1),
        )
        source_counts = np.bincount(
            indices.astype(np.int64), minlength=SCENE_HISTOGRAM_SOURCE_BINS,
        ).astype(np.uint64, copy=False)
    merged = source_counts.reshape(SCENE_HISTOGRAM_BINS, 2).sum(
        axis=1, dtype=np.uint64)
    return merged, float(raw_min32), float(raw_max32)


def _otsu_three_class_split(counts: np.ndarray) -> Optional[Tuple[int, int, float]]:
    """Return the lexicographically first valid three-class Otsu split.

    Score arithmetic intentionally follows the one-thread float32 shader: normalized bin centers,
    ascending moment accumulation, inclusive class endpoints, and a strict ``>`` update.  The
    strict comparison makes equal-score plateaus deterministic rather than dependent on a later
    traversal or platform sort.
    """

    histogram = np.asarray(counts, dtype=np.uint64)
    if histogram.shape != (SCENE_HISTOGRAM_BINS,):
        raise ValueError("scene histogram must contain exactly 128 bins")
    total_count = int(np.sum(histogram, dtype=np.uint64))
    if total_count <= 0:
        return None
    centers = np.asarray(
        [(np.float32(index) + np.float32(0.5)) / np.float32(SCENE_HISTOGRAM_BINS)
         for index in range(SCENE_HISTOGRAM_BINS)],
        dtype=np.float32,
    )
    total_moment = np.float32(0.0)
    for index in range(SCENE_HISTOGRAM_BINS):
        total_moment = np.float32(
            total_moment + np.float32(histogram[index]) * centers[index])
    global_mean = np.float32(total_moment / np.float32(total_count))

    best_score = np.float32(-1.0)
    best: Optional[Tuple[int, int, float]] = None
    far_count = 0
    far_moment = np.float32(0.0)
    minimum_mass_numerator = total_count
    for lower in range(SCENE_HISTOGRAM_BINS - 2):
        far_count += int(histogram[lower])
        far_moment = np.float32(
            far_moment + np.float32(histogram[lower]) * centers[lower])
        if far_count * SCENE_HISTOGRAM_BINS < minimum_mass_numerator:
            continue
        middle_count = 0
        middle_moment = np.float32(0.0)
        for upper in range(lower + 1, SCENE_HISTOGRAM_BINS - 1):
            middle_count += int(histogram[upper])
            middle_moment = np.float32(
                middle_moment + np.float32(histogram[upper]) * centers[upper])
            near_count = total_count - far_count - middle_count
            if (middle_count * SCENE_HISTOGRAM_BINS < minimum_mass_numerator or
                    near_count * SCENE_HISTOGRAM_BINS < minimum_mass_numerator):
                continue
            near_moment = np.float32(total_moment - far_moment - middle_moment)
            far_mean = np.float32(far_moment / np.float32(far_count))
            middle_mean = np.float32(middle_moment / np.float32(middle_count))
            near_mean = np.float32(near_moment / np.float32(near_count))
            far_delta = np.float32(far_mean - global_mean)
            middle_delta = np.float32(middle_mean - global_mean)
            near_delta = np.float32(near_mean - global_mean)
            far_term = np.float32(
                np.float32(far_count) * np.float32(far_delta * far_delta))
            middle_term = np.float32(
                np.float32(middle_count) * np.float32(middle_delta * middle_delta))
            near_term = np.float32(
                np.float32(near_count) * np.float32(near_delta * near_delta))
            score = np.float32(np.float32(far_term + middle_term) + near_term)
            if score > best_score:
                best_score = score
                best = (lower, upper, float(score))
    return best


def select_scene_coordinate(
        raw_depth: np.ndarray,
        config: MappingV2Config = MappingV2Config()) -> SceneCoordinateSelection:
    """Select the scene-latched raw center; convergence stays exactly zero.

    A valid, deeply separated upper Otsu valley may move the linear/logarithmic near-stage boundary
    upward.  The rule abstains on incomplete histograms, shallow valleys, and candidates that do
    not move upward from the arithmetic mean.  Abstention is exactly the former V2 camera:
    arithmetic-mean center.  In both cases the selected center is the exact zero plane.
    """

    _validate_config(config)
    raw = _require_raw_depth(raw_depth)
    calibration = calibrate_coordinate(raw, config)
    fallback = dict(
        observed_mean=calibration.center,
        selected_center=calibration.center,
        convergence_curve=V2_DEFAULTS.convergence_curve_default,
        adopted=False,
        raw_min=calibration.raw_min,
        raw_max=calibration.raw_max,
    )
    if calibration.collapsed or calibration.raw_max <= calibration.raw_min:
        return SceneCoordinateSelection(
            **fallback, reason="collapsed-range", histogram_total=0,
            lower_split_bin=None, upper_split_bin=None, upper_split_raw=None,
            candidate_center=None, otsu_score=None, class_counts=None,
            valley_numerator=None, middle_peak=None, near_peak=None, valley_ratio=None)

    counts, histogram_min, histogram_max = _scene_histogram_128(raw)
    histogram_total = int(np.sum(counts, dtype=np.uint64))
    if histogram_total != raw.size:
        return SceneCoordinateSelection(
            **fallback, reason="incomplete-gpu-histogram",
            histogram_total=histogram_total,
            lower_split_bin=None, upper_split_bin=None, upper_split_raw=None,
            candidate_center=None, otsu_score=None, class_counts=None,
            valley_numerator=None, middle_peak=None, near_peak=None, valley_ratio=None)

    split = _otsu_three_class_split(counts)
    if split is None:
        return SceneCoordinateSelection(
            **fallback, reason="no-valid-otsu-split", histogram_total=histogram_total,
            lower_split_bin=None, upper_split_bin=None, upper_split_raw=None,
            candidate_center=None, otsu_score=None, class_counts=None,
            valley_numerator=None, middle_peak=None, near_peak=None, valley_ratio=None)
    lower, upper, score = split

    counts32 = counts.astype(np.float32)
    smoothed = np.multiply(counts32, np.float32(0.5), dtype=np.float32)
    smoothed[1:] = np.add(
        smoothed[1:],
        np.multiply(counts32[:-1], np.float32(0.25), dtype=np.float32),
        dtype=np.float32,
    )
    smoothed[:-1] = np.add(
        smoothed[:-1],
        np.multiply(counts32[1:], np.float32(0.25), dtype=np.float32),
        dtype=np.float32,
    )
    middle_peak = np.float32(np.max(smoothed[lower + 1:upper + 1]))
    near_peak = np.float32(np.max(smoothed[upper + 1:]))
    valley_numerator = np.float32(max(smoothed[upper], smoothed[upper + 1]))
    weaker_peak = np.float32(min(middle_peak, near_peak))
    valley_ratio = (np.float32(valley_numerator / weaker_peak)
                    if weaker_peak > np.float32(0.0) else np.float32(np.inf))

    raw_range = np.float32(np.float32(histogram_max) - np.float32(histogram_min))
    normalized_boundary = np.float32(
        (np.float32(upper) + np.float32(0.5)) /
        np.float32(SCENE_HISTOGRAM_BINS))
    upper_split_raw = np.float32(
        np.float32(histogram_min) + np.float32(normalized_boundary * raw_range))
    # Keep adoption identical to the production rule that previously placed the upper valley at
    # u=1. Only after that conservative predicate succeeds does the valley itself become zero.
    acceptance_center = np.float32(
        upper_split_raw - np.float32(config.raw_coordinate_scale))
    far_count = int(np.sum(counts[:lower + 1], dtype=np.uint64))
    middle_count = int(np.sum(counts[lower + 1:upper + 1], dtype=np.uint64))
    near_count = histogram_total - far_count - middle_count

    common = dict(
        **fallback,
        histogram_total=histogram_total,
        lower_split_bin=lower,
        upper_split_bin=upper,
        upper_split_raw=float(upper_split_raw),
        candidate_center=float(acceptance_center),
        otsu_score=score,
        class_counts=(far_count, middle_count, near_count),
        valley_numerator=float(valley_numerator),
        middle_peak=float(middle_peak),
        near_peak=float(near_peak),
        valley_ratio=float(valley_ratio),
    )
    if valley_ratio > np.float32(SCENE_VALLEY_RATIO_MAX):
        return SceneCoordinateSelection(**common, reason="upper-valley-not-separated")
    if not float(acceptance_center) > calibration.center:
        return SceneCoordinateSelection(**common, reason="candidate-not-above-mean")
    common.pop("selected_center")
    common.pop("convergence_curve")
    common.pop("adopted")
    return SceneCoordinateSelection(
        **common,
        selected_center=float(upper_split_raw),
        convergence_curve=V2_DEFAULTS.convergence_curve_default,
        adopted=True,
        reason="accepted-upper-stage-boundary",
    )


def asymmetric_curve(
        canonical: np.ndarray,
        config: MappingV2Config) -> np.ndarray:
    _validate_config(config)
    values = np.asarray(canonical, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("canonical depth must contain only finite values")
    resolved_near_tau = config.near_log_tau
    curved = np.empty_like(values)
    far = values < 0.0
    linear = (values >= 0.0) & (values <= 1.0)
    near = values > 1.0
    curved[far] = config.far_tau * np.expm1(values[far] / config.far_tau)
    curved[linear] = values[linear]
    excess = values[near] - 1.0
    curved[near] = 1.0 + resolved_near_tau * np.log1p(excess / resolved_near_tau)
    if not np.isfinite(curved).all():
        raise ValueError("asymmetric curve produced a non-finite value")
    return curved


def curve_relative_coordinate(
        raw_depth: np.ndarray,
        center: float,
        scale: float,
        config: MappingV2Config = MappingV2Config(),
        *,
        convergence_curve: float = V2_DEFAULTS.convergence_curve_default,
        ) -> Tuple[np.ndarray, np.ndarray]:
    raw = _require_raw_depth(raw_depth)
    if not math.isfinite(center):
        raise ValueError("coordinate center must be finite")
    _finite_positive("coordinate scale", scale)
    if not math.isfinite(convergence_curve):
        raise ValueError("convergence_curve must be finite")
    canonical = (raw - center) / scale
    curved = asymmetric_curve(canonical, config) - convergence_curve
    if not np.isfinite(canonical).all() or not np.isfinite(curved).all():
        raise ValueError("shot calibration produced a non-finite coordinate")
    return canonical, curved


def pointwise_soft_container(
        requested: np.ndarray,
        limit: float = DIRECT_PARALLAX_SOURCE_U_LIMIT) -> np.ndarray:
    """Apply the live fourth-order soft representation bound independently per texel."""

    values = np.asarray(requested, dtype=np.float64)
    if np.iscomplexobj(requested) or not np.isfinite(values).all():
        raise ValueError("requested parallax must be finite and real-valued")
    bound = _finite_positive("limit", limit)
    magnitudes = np.abs(values)
    smaller = np.minimum(magnitudes, bound)
    larger = np.maximum(magnitudes, bound)
    ratio = smaller / larger
    ratio_squared = ratio * ratio
    fourth_root = np.sqrt(np.sqrt(1.0 + ratio_squared * ratio_squared))
    contained = np.copysign(smaller / fourth_root, values)
    return np.clip(contained, -bound, bound)


def horizontal_lipschitz_majorant(field: np.ndarray, max_step: float) -> np.ndarray:
    values = _require_raw_depth(field)
    step = _finite_positive("max_step", max_step)
    limited = values.copy()
    for x in range(1, limited.shape[1]):
        limited[:, x] = np.maximum(limited[:, x], limited[:, x - 1] - step)
    for x in range(limited.shape[1] - 2, -1, -1):
        limited[:, x] = np.maximum(limited[:, x], limited[:, x + 1] - step)
    if not np.isfinite(limited).all():
        raise ValueError("horizontal limiter produced a non-finite value")
    return limited


def vertical_lipschitz_majorant(field: np.ndarray, max_step: float) -> np.ndarray:
    """Least column-wise near-preserving majorant for an aspect-matched depth grid.

    Parallax is expressed in source-U.  For DAV2 grids fitted to the source aspect ratio,
    ``max_vertical_shear / depth_width`` therefore bounds horizontal source-pixel displacement
    change per vertical source pixel.  The authenticated model calibration owns that
    aspect-matched-grid assumption.
    """

    values = _require_raw_depth(field)
    step = _finite_positive("max_step", max_step)
    limited = values.copy()
    for y in range(1, limited.shape[0]):
        limited[y, :] = np.maximum(limited[y, :], limited[y - 1, :] - step)
    for y in range(limited.shape[0] - 2, -1, -1):
        limited[y, :] = np.maximum(limited[y, :], limited[y + 1, :] - step)
    if not np.isfinite(limited).all():
        raise ValueError("vertical limiter produced a non-finite value")
    return limited


def vertical_lipschitz_minorant(field: np.ndarray, max_step: float) -> np.ndarray:
    """Greatest column-wise lower envelope under the authenticated vertical-shear bound."""

    values = _require_raw_depth(field)
    step = _finite_positive("max_step", max_step)
    limited = values.copy()
    for y in range(1, limited.shape[0]):
        limited[y, :] = np.minimum(limited[y, :], limited[y - 1, :] + step)
    for y in range(limited.shape[0] - 2, -1, -1):
        limited[y, :] = np.minimum(limited[y, :], limited[y + 1, :] + step)
    if not np.isfinite(limited).all():
        raise ValueError("vertical lower envelope produced a non-finite value")
    return limited


def _max_adjacent_step(field: np.ndarray) -> float:
    return 0.0 if field.shape[1] < 2 else float(np.max(np.abs(np.diff(field, axis=1))))


def _max_adjacent_vertical_step(field: np.ndarray) -> float:
    return 0.0 if field.shape[0] < 2 else float(np.max(np.abs(np.diff(field, axis=0))))


def _collar_texels(max_adjacent_step: float, max_step: float) -> int:
    return 0 if max_adjacent_step <= 0.0 else int(
        math.ceil(max_adjacent_step / max_step - 1.0e-12))


def _canonical_quantiles(canonical: np.ndarray) -> Tuple[float, float, float]:
    values = np.quantile(canonical, (0.01, 0.50, 0.99))
    return float(values[0]), float(values[1]), float(values[2])


def encode_direct_parallax(
        parallax: np.ndarray,
        limit: float = DIRECT_PARALLAX_SOURCE_U_LIMIT) -> np.ndarray:
    values = _require_raw_depth(parallax)
    bound = _finite_positive("limit", limit)
    tolerance = max(1.0e-12, bound * 1.0e-7)
    maximum = float(np.max(np.abs(values)))
    if maximum > bound + tolerance:
        raise ValueError(f"direct parallax exceeds the source-U limit: {maximum} > {bound}")
    encoded = 0.5 + values / (2.0 * bound)
    encoded32 = encoded.astype("<f4")
    if not np.isfinite(encoded32).all() or np.any(encoded32 < 0.0) or np.any(encoded32 > 1.0):
        raise ValueError("direct parallax encoding is outside finite float32 [0,1]")
    return encoded32


def decode_direct_parallax(
        encoded: np.ndarray,
        limit: float = DIRECT_PARALLAX_SOURCE_U_LIMIT) -> np.ndarray:
    values = _require_raw_depth(encoded)
    bound = _finite_positive("limit", limit)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("encoded direct parallax must lie in [0,1]")
    decoded = ((values * 2.0 - 1.0) * bound).astype(np.float32)
    if not np.isfinite(decoded).all():
        raise ValueError("decoded direct parallax is not finite float32")
    return decoded


def generate_depth_mapping_v2(
        raw_depth: np.ndarray,
        config: MappingV2Config = MappingV2Config()) -> MappingV2Result:
    _validate_config(config)
    raw = _require_raw_depth(raw_depth)
    calibration = calibrate_coordinate(raw, config)
    scene_coordinate = select_scene_coordinate(raw, config)
    width = raw.shape[1]
    max_step = config.max_horizontal_slope / width
    max_vertical_step = config.max_vertical_shear / width
    convergence_curve = scene_coordinate.convergence_curve

    if calibration.collapsed:
        zero = np.zeros(raw.shape, dtype=np.float32)
        diagnostics = MappingV2Diagnostics(
            shape=raw.shape,
            center_mean=calibration.center,
            selected_center=scene_coordinate.selected_center,
            scene_coordinate=scene_coordinate,
            observed_std=calibration.observed_std,
            raw_coordinate_scale=config.raw_coordinate_scale,
            collapse_threshold=config.collapse_abs_epsilon,
            collapsed=True,
            canonical_min=0.0, canonical_p01=0.0, canonical_p50=0.0,
            canonical_p99=0.0, canonical_max=0.0,
            convergence_curve=convergence_curve,
            curve_far_limit=-config.far_tau - convergence_curve, curve_near_limit=None,
            requested_gain=config.parallax_gain, effective_gain=0.0,
            container_source_u_limit=config.direct_container_limit,
            container_scale=1.0,
            max_horizontal_slope=config.max_horizontal_slope,
            max_step_per_texel=max_step,
            max_vertical_shear=config.max_vertical_shear,
            max_vertical_step_per_texel=max_vertical_step,
            max_adjacent_step_before_container=0.0,
            max_adjacent_step_after_container=0.0,
            max_adjacent_step_after_limiter=0.0,
            max_adjacent_vertical_step_before_limiter=0.0,
            max_adjacent_vertical_step_after_vertical_limiter=0.0,
            max_adjacent_vertical_step_after_horizontal_limiter=0.0,
            estimated_collar_texels_before_container=0,
            estimated_collar_texels_after_container=0,
            requested_min=0.0, requested_max=0.0, output_min=0.0, output_max=0.0,
            vertical_conditioned_raised_fraction=0.0,
            vertical_conditioned_max_raise=0.0,
            vertical_conditioned_lowered_fraction=0.0,
            vertical_conditioned_max_lower=0.0,
            horizontal_limiter_raised_fraction=0.0, horizontal_limiter_max_raise=0.0,
            horizontal_limiter_illegal_lower_count=0,
            conditioner_raised_fraction=0.0, conditioner_max_raise=0.0,
            conditioner_lowered_fraction=0.0, conditioner_max_lower=0.0,
        )
        return MappingV2Result(
            zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero, diagnostics)

    canonical = (raw - scene_coordinate.selected_center) / calibration.scale
    curve_relative = asymmetric_curve(canonical, config) - convergence_curve
    requested = curve_relative * config.parallax_gain
    container_scale = 1.0
    conditioned = pointwise_soft_container(requested, config.direct_container_limit)
    vertical_majorant = vertical_lipschitz_majorant(conditioned, max_vertical_step)
    vertical_minorant = vertical_lipschitz_minorant(conditioned, max_vertical_step)
    # The contract is consumed by a float32 HLSL shader. Canonicalize both coefficients to the
    # exact values used there rather than rejecting ordinary decimal policy values such as 0.7.
    # The surrounding NumPy path remains a float64 semantic oracle; the executable GPU test owns
    # bit-exact per-operation parity.
    majorant_share, minorant_share = vertical_share_coefficients(
        config.vertical_majorant_share)
    vertical_limited = (
        majorant_share * vertical_majorant +
        minorant_share * vertical_minorant)
    limited = horizontal_lipschitz_majorant(vertical_limited, max_step)
    tolerance = max(1.0e-12, max(max_step, max_vertical_step) * 1.0e-9)
    if _max_adjacent_vertical_step(vertical_limited) > max_vertical_step + tolerance:
        raise RuntimeError("vertical limiter failed its Lipschitz contract")
    if _max_adjacent_step(limited) > max_step + tolerance:
        raise RuntimeError("horizontal limiter failed its Lipschitz contract")
    if _max_adjacent_vertical_step(limited) > max_vertical_step + tolerance:
        raise RuntimeError("horizontal limiter violated the vertical shear contract")
    vertical_lowered = vertical_limited < conditioned - tolerance
    horizontal_lowered = limited < vertical_limited - tolerance
    lowered = limited < conditioned - tolerance
    if np.any(horizontal_lowered):
        raise RuntimeError("row majorant lowered the vertically conditioned field")
    vertical_correction = vertical_limited - conditioned
    horizontal_raise = limited - vertical_limited
    correction = limited - conditioned
    p01, p50, p99 = _canonical_quantiles(canonical)
    diagnostics = MappingV2Diagnostics(
        shape=raw.shape,
        center_mean=calibration.center,
        selected_center=scene_coordinate.selected_center,
        scene_coordinate=scene_coordinate,
        observed_std=calibration.observed_std,
        raw_coordinate_scale=config.raw_coordinate_scale,
        collapse_threshold=config.collapse_abs_epsilon,
        collapsed=False,
        canonical_min=float(np.min(canonical)), canonical_p01=p01,
        canonical_p50=p50, canonical_p99=p99, canonical_max=float(np.max(canonical)),
        convergence_curve=convergence_curve,
        curve_far_limit=-config.far_tau - convergence_curve, curve_near_limit=None,
        requested_gain=config.parallax_gain,
        effective_gain=config.parallax_gain,
        container_source_u_limit=config.direct_container_limit,
        container_scale=container_scale,
        max_horizontal_slope=config.max_horizontal_slope,
        max_step_per_texel=max_step,
        max_vertical_shear=config.max_vertical_shear,
        max_vertical_step_per_texel=max_vertical_step,
        max_adjacent_step_before_container=_max_adjacent_step(requested),
        max_adjacent_step_after_container=_max_adjacent_step(conditioned),
        max_adjacent_step_after_limiter=_max_adjacent_step(limited),
        max_adjacent_vertical_step_before_limiter=_max_adjacent_vertical_step(conditioned),
        max_adjacent_vertical_step_after_vertical_limiter=(
            _max_adjacent_vertical_step(vertical_limited)),
        max_adjacent_vertical_step_after_horizontal_limiter=(
            _max_adjacent_vertical_step(limited)),
        estimated_collar_texels_before_container=_collar_texels(
            _max_adjacent_step(requested), max_step),
        estimated_collar_texels_after_container=_collar_texels(
            _max_adjacent_step(conditioned), max_step),
        requested_min=float(np.min(requested)), requested_max=float(np.max(requested)),
        output_min=float(np.min(limited)), output_max=float(np.max(limited)),
        vertical_conditioned_raised_fraction=float(
            np.mean(vertical_correction > tolerance)),
        vertical_conditioned_max_raise=max(0.0, float(np.max(vertical_correction))),
        vertical_conditioned_lowered_fraction=float(np.mean(vertical_lowered)),
        vertical_conditioned_max_lower=max(0.0, float(np.max(-vertical_correction))),
        horizontal_limiter_raised_fraction=float(np.mean(horizontal_raise > tolerance)),
        horizontal_limiter_max_raise=float(np.max(horizontal_raise)),
        horizontal_limiter_illegal_lower_count=int(np.count_nonzero(horizontal_lowered)),
        conditioner_raised_fraction=float(np.mean(correction > tolerance)),
        conditioner_max_raise=max(0.0, float(np.max(correction))),
        conditioner_lowered_fraction=float(np.mean(lowered)),
        conditioner_max_lower=max(0.0, float(np.max(-correction))),
    )
    outputs = tuple(field.astype(np.float32) for field in (
        canonical, requested, conditioned, vertical_limited, limited))
    if not all(np.isfinite(field).all() for field in outputs):
        raise ValueError("mapping cannot be represented by finite float32 fields")
    return MappingV2Result(*outputs, diagnostics)


__all__ = [
    "DIRECT_PARALLAX_SOURCE_U_LIMIT",
    "CoordinateCalibration",
    "MappingV2Config",
    "MappingV2Diagnostics",
    "MappingV2Result",
    "SCENE_HISTOGRAM_BINS",
    "SCENE_HISTOGRAM_SOURCE_BINS",
    "SCENE_VALLEY_RATIO_MAX",
    "SceneCoordinateSelection",
    "asymmetric_curve",
    "calibrate_coordinate",
    "curve_relative_coordinate",
    "decode_direct_parallax",
    "encode_direct_parallax",
    "generate_depth_mapping_v2",
    "horizontal_lipschitz_majorant",
    "pointwise_soft_container",
    "select_scene_coordinate",
    "vertical_lipschitz_majorant",
    "vertical_lipschitz_minorant",
]
