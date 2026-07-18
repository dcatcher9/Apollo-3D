"""Exact-map-registered interocular exposure and colour-gain rivalry metrics.

Stereo output can preserve geometry and detail while one eye is brighter or has a different
channel gain.  Such a mismatch causes binocular rivalry, yet a mono-source residual or a generic
whole-image quality score cannot distinguish it from legitimate source colour.  This module
registers both rendered eyes onto the same source-coordinate raster using the production inverse
maps, removes the analytically known clean-render residual independently for each eye, then
compares their *source-relative* photometric changes.

Two axes are deliberately kept separate:

``interocular_exposure_rivalry_burden_pct``
    Coherent RMS excess of the inter-eye log-luminance-ratio disagreement.  For a sufficiently
    bright global transform, a unilateral gain ``g`` is approximately ``100 * abs(log(g))``;
    the same transform applied to both eyes cancels.

``interocular_color_gain_rivalry_burden_pct``
    Coherent RMS excess in two orthonormal log-chroma opponent coordinates.  It catches
    unilateral white-balance, RGB-gain and hue changes independently of exposure.  It is not a
    Delta-E value and must not be averaged with the exposure axis.

Only mutually visible, locally unique exact-map samples vote.  Optional forward-hole masks
exclude inpainted/disoccluded pixels whose final colour has no authenticated mono-source
correspondence.  The module returns ``None`` for a severity when support is insufficient and
exposes registered maps with ``return_maps=True`` for visual qualification.
"""

from __future__ import annotations

import numpy as np

try:
    from . import sbs_interocular_metrics as _registration
    from . import sbs_interocular_phase_chroma as _exact
except ImportError:  # Direct execution from tools/sbsbench.
    import sbs_interocular_metrics as _registration
    import sbs_interocular_phase_chroma as _exact


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
_SQRT_2 = np.sqrt(2.0)
_SQRT_6 = np.sqrt(6.0)


def _linearize_srgb(rgb):
    """Convert non-negative encoded sRGB to linear light, preserving values above one."""
    value = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _split_hole_masks(warp_mask, eye_shape):
    """Return optional left/right float hole maps from a packed or paired mask."""
    if warp_mask is None:
        return None, None
    height, width = eye_shape
    if isinstance(warp_mask, (tuple, list)) and len(warp_mask) == 2:
        values = tuple(np.asarray(item) for item in warp_mask)
    else:
        packed = np.asarray(warp_mask)
        if packed.ndim not in (2, 3) or packed.shape[:2] != (height, 2 * width):
            raise ValueError(
                f"packed warp_mask must be {height}x{2 * width}[xC], got {packed.shape}")
        values = packed[:, :width], packed[:, width:]

    result = []
    for value in values:
        if value.ndim == 3:
            if value.shape[2] < 1:
                raise ValueError("warp_mask has no red channel")
            value = value[..., 0]
        if value.shape != (height, width):
            raise ValueError(
                f"eye warp_mask must be {height}x{width}, got {value.shape}")
        value = value.astype(np.float32)
        if not np.isfinite(value).all():
            raise ValueError("warp_mask contains non-finite values")
        result.append(value)
    return tuple(result)


def _opponent_coordinates(rgb, floor):
    """Return log luminance and orthonormal log-chroma in linear-light RGB."""
    linear = _linearize_srgb(rgb)
    luminance = linear @ _LUMA
    log_luminance = np.log(np.maximum(luminance, floor))
    log_rgb = np.log(np.maximum(linear, floor))
    first = (log_rgb[..., 0] - log_rgb[..., 1]) / _SQRT_2
    second = (log_rgb[..., 0] + log_rgb[..., 1]
              - 2.0 * log_rgb[..., 2]) / _SQRT_6
    return log_luminance, np.stack((first, second), axis=2), luminance


def _registered_evidence(source, eye, source_u_map, shape, width, height,
                         source_sample_transform, hole_mask):
    expected_eye = _exact._apply_sample_transform(
        _exact._sample_source_eye(source, source_u_map, shape),
        source_sample_transform, eye.shape)
    channels = [eye, expected_eye]
    if hole_mask is not None:
        channels.append(hole_mask[..., None])
    registered, valid = _exact._register_rgb(
        np.concatenate(channels, axis=2), source_u_map, shape, width, height)
    actual = registered[..., :3]
    expected = registered[..., 3:6]
    if hole_mask is not None:
        # A bilinear footprint touching an unauthenticated forward hole is excluded.  Erosion
        # below removes one more analysis pixel so mask edges cannot leak into colour evidence.
        valid &= registered[..., 6] <= 1e-6
    return actual, expected, valid


def _erode(mask, radius=1):
    return _registration._box_mean(
        np.asarray(mask, dtype=np.float32), int(radius)) > 1.0 - 1e-6


def _native_equivalent_count(mask, evidence_basis):
    return int(round(np.count_nonzero(mask) / max(mask.size, 1) * evidence_basis))


def _connected_components(mask):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [int(start_y * width + start_x)]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            y, x = divmod(index, width)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < height and 0 <= nx < width and mask[ny, nx]
                        and not visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append(ny * width + nx)
        yield np.asarray(component, dtype=np.int64)


def _localized_burden(values, weights, support, visibility_floor, max_components=8):
    """Pool coherent severity without a percentile footprint blind spot."""
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    qualified = support & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(qualified):
        return None
    total_weight = float(np.sum(weights[qualified]))
    if total_weight <= 1e-12:
        return None
    excess = np.maximum(values - float(visibility_floor), 0.0)
    active = qualified & (excess > 0.0)
    flat_excess, flat_weight = excess.ravel(), weights.ravel()
    energies = [
        float(np.sum(flat_weight[component] * np.square(flat_excess[component])))
        for component in _connected_components(active)
    ]
    strongest = sorted(energies, reverse=True)[:int(max_components)]
    return float(np.sqrt(sum(strongest) / total_weight)) if strongest else 0.0


def measure_interocular_photometric_rivalry(
        source_rgb, left_rgb, right_rgb, map_left, map_right, shape, *,
        warp_mask=None, source_sample_transform=None,
        max_analysis_width=640, max_analysis_height=360, min_pixels=256,
        return_maps=False):
    """Measure source-relative exposure and colour-gain disagreement between SBS eyes.

    ``warp_mask`` may be the packed SBS mask or ``(left_mask, right_mask)``.  Its red channel is
    interpreted as the harness forward-hole flag.  ``source_sample_transform`` has the same
    production-order contract as the exact phase/orientation metric: it is applied after fractional
    source sampling to both canonical and expected-eye references.
    """
    source = _exact._as_rgb(source_rgb, "source_rgb")
    left = _exact._as_rgb(left_rgb, "left_rgb")
    right = _exact._as_rgb(right_rgb, "right_rgb")
    if left.shape != right.shape:
        raise ValueError(f"eye geometry differs: {left.shape} != {right.shape}")
    if np.asarray(map_left).shape != left.shape[:2]:
        raise ValueError("map_left must match the left eye geometry")
    if np.asarray(map_right).shape != right.shape[:2]:
        raise ValueError("map_right must match the right eye geometry")
    hole_left, hole_right = _split_hole_masks(warp_mask, left.shape[:2])

    analysis_scale = min(
        float(max_analysis_width) / source.shape[1],
        float(max_analysis_height) / source.shape[0],
    )
    analysis_width = max(16, int(round(source.shape[1] * analysis_scale)))
    analysis_height = max(12, int(round(source.shape[0] * analysis_scale)))
    reference = _exact._apply_sample_transform(
        _exact._sample_source_rgb(source, analysis_width, analysis_height),
        source_sample_transform, (analysis_height, analysis_width, 3))
    left_actual, left_expected, left_valid = _registered_evidence(
        source, left, np.asarray(map_left, dtype=np.float32), shape,
        analysis_width, analysis_height, source_sample_transform, hole_left)
    right_actual, right_expected, right_valid = _registered_evidence(
        source, right, np.asarray(map_right, dtype=np.float32), shape,
        analysis_width, analysis_height, source_sample_transform, hole_right)
    common = _erode(left_valid & right_valid, 1)

    content_samples = int(round(
        left.shape[0] * left.shape[1] * float(shape["content_scale_x"])
        * float(shape["content_scale_y"])))
    evidence_basis = max(1, min(source.shape[0] * source.shape[1], content_samples))

    metrics = {
        "interocular_exposure_rivalry_burden_pct": None,
        "interocular_exposure_rivalry_support_pct": 0.0,
        "interocular_exposure_rivalry_support_count": 0,
        "interocular_exposure_rivalry_evidence_sufficient": 0.0,
        "interocular_color_gain_rivalry_burden_pct": None,
        "interocular_color_gain_rivalry_support_pct": 0.0,
        "interocular_color_gain_rivalry_support_count": 0,
        "interocular_color_gain_rivalry_evidence_sufficient": 0.0,
    }

    reference_linear = _linearize_srgb(reference)
    reference_luminance = reference_linear @ _LUMA
    common_luminance = reference_luminance[common]
    brightness_scale = (float(np.quantile(common_luminance, 0.95))
                        if common_luminance.size else 0.0)
    # Ignore deep encoded shadows where one 8-bit code step creates a huge relative ratio.
    brightness_floor = max(brightness_scale * 0.01, _linearize_srgb(2.0 / 255.0))
    support = common & (reference_luminance >= float(brightness_floor))
    support_count = _native_equivalent_count(support, evidence_basis)
    support_pct = float(100.0 * np.count_nonzero(support) / max(support.size, 1))
    required = max(int(min_pixels), int(np.ceil(evidence_basis * 0.005)))

    floor = max(float(brightness_floor) / 16.0, 1e-8)
    left_luma, left_chroma, _ = _opponent_coordinates(left_actual, floor)
    right_luma, right_chroma, _ = _opponent_coordinates(right_actual, floor)
    expected_left_luma, expected_left_chroma, _ = _opponent_coordinates(
        left_expected, floor)
    expected_right_luma, expected_right_chroma, _ = _opponent_coordinates(
        right_expected, floor)

    # Each eye votes with its change relative to the exact clean render.  Subtracting those
    # changes cancels a shared exposure/RGB transform while retaining a one-eye transform.
    exposure_map = 100.0 * np.abs(
        (left_luma - expected_left_luma) - (right_luma - expected_right_luma))
    color_map = 100.0 * np.linalg.norm(
        (left_chroma - expected_left_chroma)
        - (right_chroma - expected_right_chroma), axis=2)
    weight = np.sqrt(np.maximum(reference_luminance, 0.0))
    if np.any(support):
        weight_cap = float(np.quantile(weight[support], 0.95))
        weight = np.minimum(weight, weight_cap)
    else:
        weight = np.zeros_like(weight)

    for prefix in ("interocular_exposure_rivalry", "interocular_color_gain_rivalry"):
        metrics[f"{prefix}_support_pct"] = support_pct
        metrics[f"{prefix}_support_count"] = support_count
    if support_count >= required and brightness_scale >= _linearize_srgb(4.0 / 255.0):
        metrics["interocular_exposure_rivalry_evidence_sufficient"] = 100.0
        metrics["interocular_color_gain_rivalry_evidence_sufficient"] = 100.0
        metrics["interocular_exposure_rivalry_burden_pct"] = _localized_burden(
            exposure_map, weight, support, visibility_floor=1.0)
        metrics["interocular_color_gain_rivalry_burden_pct"] = _localized_burden(
            color_map, weight, support, visibility_floor=1.5)

    if not return_maps:
        return metrics
    return metrics, {
        "registered_source_rgb": reference,
        "registered_left_rgb": left_actual,
        "registered_right_rgb": right_actual,
        "expected_left_rgb": left_expected,
        "expected_right_rgb": right_expected,
        "common_support": common,
        "photometric_support": support,
        "exposure_rivalry_pct": np.where(support, exposure_map, np.nan),
        "color_gain_rivalry_pct": np.where(support, color_map, np.nan),
    }


__all__ = ["measure_interocular_photometric_rivalry"]
