#!/usr/bin/env python3
"""NumPy comparison oracle for the production depth-coordinate-v2 GPU mapping.

The camera center is acquired from the first usable DAV2 field. Scale is a fixed authenticated
model/shape calibration and never adapts to a frame's distribution. A separately named
curve-space convergence value is latched with the camera and is currently exactly zero. The
near-tail curve is selected from scene-acquisition occupancy and then held with that camera. Only
the hard representation container adapts per frame; requested gain never becomes mutable shot
state, and its extrema are deliberately evaluated with the conservative base near curve.
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


@dataclass(frozen=True)
class MappingV2Config:
    raw_coordinate_scale: float = V2_MODEL_CALIBRATIONS[0].raw_coordinate_scale
    collapse_abs_epsilon: float = V2_DEFAULTS.collapse_abs_epsilon
    far_tau: float = V2_DEFAULTS.far_tau
    near_log_tau: float = V2_DEFAULTS.near_log_tau
    near_tail_probe_u: float = V2_DEFAULTS.near_tail_probe_u
    near_tail_coverage_low: float = V2_DEFAULTS.near_tail_coverage_low
    near_tail_coverage_high: float = V2_DEFAULTS.near_tail_coverage_high
    near_log_tau_dense: float = V2_DEFAULTS.near_log_tau_dense
    pop_strength: float = V2_DEFAULTS.reference_pop_strength
    gain_per_pop: float = V2_DEFAULTS.gain_per_pop
    max_horizontal_slope: float = V2_DEFAULTS.max_horizontal_slope
    max_vertical_shear: float = V2_DEFAULTS.max_vertical_shear
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
class MappingV2Diagnostics:
    shape: Tuple[int, int]
    center_mean: float
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
    near_tail_coverage: float
    dense_near_weight: float
    effective_near_log_tau: float
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
    vertical_limiter_raised_fraction: float
    vertical_limiter_max_raise: float
    vertical_limiter_illegal_lower_count: int
    horizontal_limiter_raised_fraction: float
    horizontal_limiter_max_raise: float
    horizontal_limiter_illegal_lower_count: int
    limiter_raised_fraction: float
    limiter_max_raise: float
    limiter_illegal_lower_count: int

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
    _finite_positive("near_tail_probe_u", config.near_tail_probe_u)
    if config.near_tail_probe_u < 1.0:
        raise ValueError("near_tail_probe_u must be at least the near-curve knee")
    for name, value in (
            ("near_tail_coverage_low", config.near_tail_coverage_low),
            ("near_tail_coverage_high", config.near_tail_coverage_high)):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and lie in [0, 1]")
    if config.near_tail_coverage_low >= config.near_tail_coverage_high:
        raise ValueError("near-tail coverage thresholds must be strictly ordered")
    _finite_positive("near_log_tau_dense", config.near_log_tau_dense)
    if config.near_log_tau_dense >= config.near_log_tau:
        raise ValueError("near_log_tau_dense must be below near_log_tau")
    _finite_positive("pop_strength", config.pop_strength)
    _finite_positive("gain_per_pop", config.gain_per_pop)
    _finite_positive("max_horizontal_slope", config.max_horizontal_slope)
    if config.max_horizontal_slope >= 1.0:
        raise ValueError("max_horizontal_slope must be below 1 for a no-fold forward map")
    _finite_positive("max_vertical_shear", config.max_vertical_shear)
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


def near_tail_count_and_coverage(
        canonical: np.ndarray,
        config: MappingV2Config = MappingV2Config()) -> Tuple[int, float]:
    """Measure canonical near-tail occupancy using the authenticated scene probe.

    The strict ``>`` comparison intentionally matches the GPU reduction. A value at the linear
    knee is not part of the compressed logarithmic tail.
    """

    _validate_config(config)
    values = np.asarray(canonical, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("canonical depth must be finite and non-empty")
    count = int(np.count_nonzero(values > config.near_tail_probe_u))
    return count, count / values.size


def dense_near_weight(
        coverage: float,
        config: MappingV2Config = MappingV2Config()) -> float:
    """Return the smoothstep weight selecting the dense near-tail curve."""

    _validate_config(config)
    if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError("near-tail coverage must be finite and lie in [0, 1]")
    position = ((coverage - config.near_tail_coverage_low) /
                (config.near_tail_coverage_high - config.near_tail_coverage_low))
    clamped = min(max(position, 0.0), 1.0)
    return clamped * clamped * (3.0 - 2.0 * clamped)


def effective_near_log_tau(
        coverage: float,
        config: MappingV2Config = MappingV2Config()) -> float:
    """Interpolate from the base to dense-tail tau for one latched scene camera."""

    weight = dense_near_weight(coverage, config)
    return config.near_log_tau + weight * (
        config.near_log_tau_dense - config.near_log_tau)


def asymmetric_curve(
        canonical: np.ndarray,
        config: MappingV2Config,
        *,
        near_log_tau: Optional[float] = None) -> np.ndarray:
    _validate_config(config)
    values = np.asarray(canonical, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("canonical depth must contain only finite values")
    resolved_near_tau = (config.near_log_tau if near_log_tau is None else
                         _finite_positive("effective near_log_tau", near_log_tau))
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
        near_log_tau: Optional[float] = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    raw = _require_raw_depth(raw_depth)
    if not math.isfinite(center):
        raise ValueError("coordinate center must be finite")
    _finite_positive("coordinate scale", scale)
    if not math.isfinite(convergence_curve):
        raise ValueError("convergence_curve must be finite")
    canonical = (raw - center) / scale
    curved = asymmetric_curve(
        canonical, config, near_log_tau=near_log_tau) - convergence_curve
    if not np.isfinite(canonical).all() or not np.isfinite(curved).all():
        raise ValueError("shot calibration produced a non-finite coordinate")
    return canonical, curved


def container_scale_for_curve_range(
        curve_min: float,
        curve_max: float,
        config: MappingV2Config = MappingV2Config(),
        *,
        gain: Optional[float] = None,
        ) -> Tuple[float, float]:
    _validate_config(config)
    resolved_gain = config.parallax_gain if gain is None else _finite_positive("gain", gain)
    if not math.isfinite(curve_min) or not math.isfinite(curve_max) or curve_min > curve_max:
        raise ValueError("curve range must be finite and ordered")
    requested_maximum = resolved_gain * max(abs(curve_min), abs(curve_max))
    container_scale = 1.0 if requested_maximum <= config.direct_container_limit else (
        config.direct_container_limit / requested_maximum)
    return container_scale, requested_maximum


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
    width = raw.shape[1]
    max_step = config.max_horizontal_slope / width
    max_vertical_step = config.max_vertical_shear / width
    convergence_curve = V2_DEFAULTS.convergence_curve_default
    tail_coverage = 0.0
    tail_weight = 0.0
    scene_near_tau = config.near_log_tau

    if calibration.collapsed:
        zero = np.zeros(raw.shape, dtype=np.float32)
        diagnostics = MappingV2Diagnostics(
            shape=raw.shape,
            center_mean=calibration.center,
            observed_std=calibration.observed_std,
            raw_coordinate_scale=config.raw_coordinate_scale,
            collapse_threshold=config.collapse_abs_epsilon,
            collapsed=True,
            canonical_min=0.0, canonical_p01=0.0, canonical_p50=0.0,
            canonical_p99=0.0, canonical_max=0.0,
            convergence_curve=convergence_curve,
            near_tail_coverage=tail_coverage,
            dense_near_weight=tail_weight,
            effective_near_log_tau=scene_near_tau,
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
            vertical_limiter_raised_fraction=0.0, vertical_limiter_max_raise=0.0,
            vertical_limiter_illegal_lower_count=0,
            horizontal_limiter_raised_fraction=0.0, horizontal_limiter_max_raise=0.0,
            horizontal_limiter_illegal_lower_count=0,
            limiter_raised_fraction=0.0, limiter_max_raise=0.0,
            limiter_illegal_lower_count=0,
        )
        return MappingV2Result(
            zero.copy(), zero.copy(), zero.copy(), zero.copy(), zero, diagnostics)

    canonical = (raw - calibration.center) / calibration.scale
    _, tail_coverage = near_tail_count_and_coverage(canonical, config)
    tail_weight = dense_near_weight(tail_coverage, config)
    scene_near_tau = effective_near_log_tau(tail_coverage, config)
    curve_relative = asymmetric_curve(
        canonical, config, near_log_tau=scene_near_tau) - convergence_curve
    requested = curve_relative * config.parallax_gain
    # The adaptive curve may reduce near values, but it never weakens the hard container. Compute
    # the representation envelope from the base-tau field so changing occupancy cannot authorize
    # more source-U displacement than the original V2 contract.
    base_curve_relative = asymmetric_curve(canonical, config) - convergence_curve
    container_scale, _ = container_scale_for_curve_range(
        float(np.min(base_curve_relative)), float(np.max(base_curve_relative)), config)
    conditioned = requested * container_scale
    vertical_limited = vertical_lipschitz_majorant(conditioned, max_vertical_step)
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
    if np.any(vertical_lowered) or np.any(horizontal_lowered) or np.any(lowered):
        raise RuntimeError("near-preserving two-axis limiter lowered input parallax")
    vertical_raise = vertical_limited - conditioned
    horizontal_raise = limited - vertical_limited
    raised = limited - conditioned
    p01, p50, p99 = _canonical_quantiles(canonical)
    diagnostics = MappingV2Diagnostics(
        shape=raw.shape,
        center_mean=calibration.center,
        observed_std=calibration.observed_std,
        raw_coordinate_scale=config.raw_coordinate_scale,
        collapse_threshold=config.collapse_abs_epsilon,
        collapsed=False,
        canonical_min=float(np.min(canonical)), canonical_p01=p01,
        canonical_p50=p50, canonical_p99=p99, canonical_max=float(np.max(canonical)),
        convergence_curve=convergence_curve,
        near_tail_coverage=tail_coverage,
        dense_near_weight=tail_weight,
        effective_near_log_tau=scene_near_tau,
        curve_far_limit=-config.far_tau - convergence_curve, curve_near_limit=None,
        requested_gain=config.parallax_gain,
        effective_gain=config.parallax_gain * container_scale,
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
        vertical_limiter_raised_fraction=float(np.mean(vertical_raise > tolerance)),
        vertical_limiter_max_raise=float(np.max(vertical_raise)),
        vertical_limiter_illegal_lower_count=int(np.count_nonzero(vertical_lowered)),
        horizontal_limiter_raised_fraction=float(np.mean(horizontal_raise > tolerance)),
        horizontal_limiter_max_raise=float(np.max(horizontal_raise)),
        horizontal_limiter_illegal_lower_count=int(np.count_nonzero(horizontal_lowered)),
        limiter_raised_fraction=float(np.mean(raised > tolerance)),
        limiter_max_raise=float(np.max(raised)),
        limiter_illegal_lower_count=int(np.count_nonzero(lowered)),
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
    "asymmetric_curve",
    "calibrate_coordinate",
    "container_scale_for_curve_range",
    "curve_relative_coordinate",
    "dense_near_weight",
    "decode_direct_parallax",
    "encode_direct_parallax",
    "effective_near_log_tau",
    "generate_depth_mapping_v2",
    "horizontal_lipschitz_majorant",
    "near_tail_count_and_coverage",
    "vertical_lipschitz_majorant",
]
