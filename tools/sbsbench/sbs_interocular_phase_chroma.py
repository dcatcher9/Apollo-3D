"""Experimental source-registered interocular phase/orientation metric.

Exact mapped-source conformance catches unilateral blur and missing texture, but equal retained
detail does not imply binocular compatibility: two eyes can preserve the same gradient energy
while disagreeing in local phase or orientation.

This module measures that gap directly.  It first inverts each eye's exact production
output-to-source U map onto a common source-coordinate raster.  Only mutually visible, locally
unique samples vote; bars, clamps, holes, and folded inverse-map regions are excluded.  No optical
flow or image matching is used, so legitimate stereo disparity cannot be mistaken for a defect.

``interocular_phase_orientation_burden_pct``
    A multiscale complex-gradient response disagreement.  Directed horizontal/vertical responses
    are compared after per-channel positive affine normalization to the mono source.  This makes
    global exposure and white-balance changes benign.  Support also requires comparable detail
    energy in both eyes, leaving unilateral blur to exact mapped-source image integrity instead
    of double-counting it.

The metric is experimental and deliberately has no fused score or acceptance threshold.  It
returns ``None`` when evidence is insufficient.  ``return_maps=True`` exposes the registered
evidence and masks for visual validation before the axis is promoted into the evaluator.  The
corresponding ``*_evidence_sufficient`` value is always numeric: 100 only when all dynamic
support, texture, span, and cross-eye detail-compatibility requirements pass, otherwise zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from . import sbs_interocular_metrics as _registration
except ImportError:  # Direct execution from tools/sbsbench.
    import sbs_interocular_metrics as _registration


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


@dataclass(frozen=True)
class PreparedInterocularEvidence:
    """Shared exact-map registration consumed by interocular appearance metrics.

    Preparing this evidence is substantially more expensive than either metric's final pooling:
    each eye must first be sampled at the production coordinates and then inverted onto the same
    canonical source raster.  Phase/orientation and photometric rivalry require precisely the
    same arrays, so callers evaluating both should prepare them once and use the two ``*_prepared``
    entry points below.

    Instances are created only by :func:`prepare_interocular_evidence`.  The arrays are treated as
    immutable views by consumers; retaining the native eye/source shapes makes evidence counts
    independent of the analysis raster.
    """

    reference_rgb: np.ndarray
    registered_left_rgb: np.ndarray
    registered_right_rgb: np.ndarray
    expected_left_rgb: np.ndarray
    expected_right_rgb: np.ndarray
    valid_left: np.ndarray
    valid_right: np.ndarray
    common_support: np.ndarray
    source_shape: tuple[int, int, int]
    eye_shape: tuple[int, int, int]
    analysis_width: int
    analysis_height: int
    evidence_basis: int


def _as_rgb(image, name):
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(f"{name} must be HxWx3+, got {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    value = value[..., :3]
    if value.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _sample_source_rgb(source, width, height):
    return np.stack([
        _registration._sample_source_grid(source[..., channel], width, height)
        for channel in range(3)
    ], axis=2)


def _sample_source_eye(source, source_u_map, shape):
    """Bilinearly sample raw source RGB at exact output coordinates."""
    source_u_map = np.asarray(source_u_map, dtype=np.float32)
    output_height, output_width = source_u_map.shape
    source_height, source_width = source.shape[:2]
    scale_y = float(shape.get("content_scale_y", 0.0))
    if not 0.0 < scale_y <= 1.0:
        raise ValueError("exact interocular metric requires a valid content_scale_y")
    lo_y = 0.5 * (1.0 - scale_y)
    output_v = (np.arange(output_height, dtype=np.float32) + 0.5) / output_height
    source_v = (output_v - lo_y) / scale_y

    x = np.clip(source_u_map, 0.0, 1.0) * source_width - 0.5
    y = np.clip(source_v, 0.0, 1.0) * source_height - 0.5
    x = np.clip(x, 0.0, source_width - 1.0)
    y = np.clip(y, 0.0, source_height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[:, None, None]
    rows0 = y0[:, None]
    rows1 = y1[:, None]
    top = (1.0 - fx) * source[rows0, x0] + fx * source[rows0, x1]
    bottom = (1.0 - fx) * source[rows1, x0] + fx * source[rows1, x1]
    return ((1.0 - fy) * top + fy * bottom).astype(np.float32)


def _apply_sample_transform(image, transform, expected_shape):
    if transform is None:
        return image
    transformed = np.asarray(transform(image), dtype=np.float32)
    if transformed.shape != expected_shape or not np.isfinite(transformed).all():
        raise ValueError("source_sample_transform returned invalid RGB evidence")
    return transformed


def _invert_row_channels(eye_row, source_u_row, content_x, target_u):
    """Invert one exact map row once for every image channel."""
    width, channels = target_u.size, eye_row.shape[1]
    value_sum = np.zeros((width, channels), dtype=np.float64)
    count = np.zeros(width, dtype=np.int32)
    min_position = np.full(width, np.inf, dtype=np.float64)
    max_position = np.full(width, -np.inf, dtype=np.float64)
    invertible = content_x & (source_u_row >= 0.0) & (source_u_row <= 1.0)
    for run in _registration._monotonic_runs(source_u_row, invertible):
        run_u = source_u_row[run]
        lo, hi = max(float(run_u[0]), 0.0), min(float(run_u[-1]), 1.0)
        if hi < lo:
            continue
        selected = (target_u >= lo - 1e-7) & (target_u <= hi + 1e-7)
        if not selected.any():
            continue
        targets = target_u[selected]
        for channel in range(channels):
            value_sum[selected, channel] += np.interp(
                targets, run_u, eye_row[run, channel])
        positions = np.interp(targets, run_u, run.astype(np.float32))
        count[selected] += 1
        min_position[selected] = np.minimum(min_position[selected], positions)
        max_position[selected] = np.maximum(max_position[selected], positions)

    unique = (count > 0) & ((max_position - min_position) <= 1.05)
    registered = np.full((width, channels), np.nan, dtype=np.float32)
    registered[unique] = (
        value_sum[unique] / count[unique, None]).astype(np.float32)
    return registered, unique


def _register_rgb(eye, source_u_map, shape, width, height):
    """Register N image channels while traversing the exact inverse map only once."""
    eye = np.asarray(eye, dtype=np.float32)
    source_u_map = np.asarray(source_u_map, dtype=np.float32)
    if eye.ndim != 3 or source_u_map.shape != eye.shape[:2]:
        raise ValueError("source-U map must match an HxWxC eye")
    if not np.isfinite(source_u_map).all():
        raise ValueError("source-U map contains non-finite values")

    output_height, output_width = source_u_map.shape
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if not (0.0 < scale_x <= 1.0 and 0.0 < scale_y <= 1.0):
        raise ValueError("exact interocular metric requires valid content scales")
    lo_x, lo_y = 0.5 * (1.0 - scale_x), 0.5 * (1.0 - scale_y)
    output_u = (np.arange(output_width, dtype=np.float32) + 0.5) / output_width
    content_x = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    target_u = (np.arange(width, dtype=np.float32) + 0.5) / width
    target_v = (np.arange(height, dtype=np.float32) + 0.5) / height

    registered = np.full((height, width, eye.shape[2]), np.nan, dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    for row, source_v in enumerate(target_v):
        output_y = (lo_y + source_v * scale_y) * output_height - 0.5
        eye_row = _registration._sample_output_row(eye, output_y)
        map_row = _registration._sample_output_row(source_u_map, output_y)
        registered[row], valid[row] = _invert_row_channels(
            eye_row, map_row, content_x, target_u)
    return registered, valid


def _split_hole_masks(warp_mask, eye_shape):
    """Return per-eye forward-hole masks; red/nonzero means unauthenticated fill."""
    if warp_mask is None:
        return None, None
    height, width = eye_shape
    if isinstance(warp_mask, (tuple, list)):
        if len(warp_mask) != 2:
            raise ValueError("warp_mask tuple must contain left and right eyes")
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


def _registered_evidence(source, eye, source_u_map, shape, width, height,
                         source_sample_transform, hole_mask):
    expected_eye = _apply_sample_transform(
        _sample_source_eye(source, source_u_map, shape), source_sample_transform, eye.shape)
    channels = [eye, expected_eye]
    if hole_mask is not None:
        channels.append(hole_mask[..., None])
    registered, valid = _register_rgb(
        np.concatenate(channels, axis=2), source_u_map, shape, width, height)
    actual = registered[..., :3]
    expected = registered[..., 3:6]
    if hole_mask is not None:
        valid &= registered[..., 6] <= 1e-6
    return actual, expected, valid


def _native_equivalent_count(mask, evidence_basis):
    return int(round(np.count_nonzero(mask) / max(mask.size, 1) * evidence_basis))


def _erode_support(mask, radius):
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    return _registration._box_mean(mask.astype(np.float32), radius) > 1.0 - 1e-6


def prepare_interocular_evidence(
        source_rgb, left_rgb, right_rgb, map_left, map_right, shape, *,
        warp_mask=None, source_sample_transform=None,
        max_analysis_width=640, max_analysis_height=360):
    """Register both rendered eyes and their exact references once.

    The returned evidence is valid for every interocular appearance metric using the same source,
    eyes, maps, mask, display transform, and analysis limits.  Metric-specific support selection
    happens later, so preparing once does not fuse the phase and photometric policies.
    """
    source = _as_rgb(source_rgb, "source_rgb")
    left = _as_rgb(left_rgb, "left_rgb")
    right = _as_rgb(right_rgb, "right_rgb")
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
    reference = _apply_sample_transform(
        _sample_source_rgb(source, analysis_width, analysis_height),
        source_sample_transform, (analysis_height, analysis_width, 3))
    registered_left, expected_left, valid_left = _registered_evidence(
        source, left, map_left, shape, analysis_width, analysis_height,
        source_sample_transform, hole_left)
    registered_right, expected_right, valid_right = _registered_evidence(
        source, right, map_right, shape, analysis_width, analysis_height,
        source_sample_transform, hole_right)
    common = _erode_support(valid_left & valid_right, 1)

    content_samples = int(round(
        left.shape[0] * left.shape[1] * float(shape["content_scale_x"])
        * float(shape["content_scale_y"])))
    evidence_basis = max(1, min(source.shape[0] * source.shape[1], content_samples))
    return PreparedInterocularEvidence(
        reference_rgb=reference,
        registered_left_rgb=registered_left,
        registered_right_rgb=registered_right,
        expected_left_rgb=expected_left,
        expected_right_rgb=expected_right,
        valid_left=valid_left,
        valid_right=valid_right,
        common_support=common,
        source_shape=source.shape,
        eye_shape=left.shape,
        analysis_width=analysis_width,
        analysis_height=analysis_height,
        evidence_basis=evidence_basis,
    )


def _require_prepared_evidence(prepared):
    if not isinstance(prepared, PreparedInterocularEvidence):
        raise ValueError("prepared interocular evidence has an invalid type")
    return prepared


def _central_gradients(image):
    xpad = np.pad(image, ((0, 0), (1, 1)), mode="edge")
    ypad = np.pad(image, ((1, 1), (0, 0)), mode="edge")
    return (0.5 * (xpad[:, 2:] - xpad[:, :-2]),
            0.5 * (ypad[2:, :] - ypad[:-2, :]))


def _phase_responses(image, smoothing_radius, spatial_scale, energy_radius):
    smooth = _registration._box_mean(image, smoothing_radius)
    gx, gy = _central_gradients(smooth)
    gx *= spatial_scale
    gy *= spatial_scale
    # Treat gx + i*gy as a deterministic complex-gradient response.  Its directed angle detects
    # orientation and phase/sign disagreement while remaining invariant to amplitude changes
    # from a symmetric low-pass filter.
    response = np.stack((gx, gy), axis=2)
    energy_sq = np.sum(response * response, axis=2)
    energy = np.sqrt(np.maximum(
        _registration._box_mean(energy_sq, energy_radius), 0.0))
    return response, energy


def _normalize_rgb_to_source(reference, eye, support):
    normalized = np.empty_like(eye)
    for channel in range(3):
        gain, bias = _registration._robust_affine(
            reference[..., channel], eye[..., channel], support)
        normalized[..., channel] = (eye[..., channel] - bias) / gain
    return normalized


def _connected_components(mask):
    """Yield 4-connected flat-index components from one localized conflict mask."""
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


def _localized_burden(values, weights, support, visibility_floor, max_components=4):
    """Pool coherent conflict energy without a global-percentile footprint cliff.

    A P95 is identically clean whenever a severe fault occupies less than five percent of the
    qualified raster.  Thin swords, limbs, halos, and subtitle edges commonly fall below that
    footprint.  This pooling first removes a small detector-noise floor, groups the remaining
    response into coherent four-connected regions, and reports the RMS excess energy of the four
    strongest regions relative to *all* qualified support.  Severity and affected area therefore
    both increase the result: a conflict of strength ``s`` over fraction ``a`` contributes about
    ``s * sqrt(a)`` while isolated sub-threshold noise contributes zero.

    Keeping only a bounded number of coherent components prevents thousands of scattered weak
    pixels from becoming equivalent to one visible edge artifact.  The spatial map remains the
    source of localization truth; this function only supplies its compact scalar burden.
    """
    values = np.asarray(values, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    support = np.asarray(support, dtype=bool)
    if values.shape != weights.shape or values.shape != support.shape:
        raise ValueError("localized burden inputs must have identical geometry")
    if visibility_floor < 0.0 or max_components < 1:
        raise ValueError("localized burden parameters must be non-negative")
    qualified = support & np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(qualified):
        return None
    total_weight = float(np.sum(weights[qualified]))
    if total_weight <= 1e-12:
        return None
    excess = np.maximum(values - float(visibility_floor), 0.0)
    active = qualified & (excess > 0.0)
    energies = []
    flat_excess = excess.ravel()
    flat_weight = weights.ravel()
    for component in _connected_components(active):
        energies.append(float(np.sum(
            flat_weight[component] * np.square(flat_excess[component]))))
    strongest = sorted(energies, reverse=True)[:max_components]
    return float(np.sqrt(sum(strongest) / total_weight)) if strongest else 0.0


def measure_interocular_phase_chroma(
        source_rgb, left_rgb, right_rgb, map_left, map_right, shape, *,
        warp_mask=None, source_sample_transform=None,
        max_analysis_width=640, max_analysis_height=360, min_phase_pixels=128,
        return_maps=False):
    """Measure phase/orientation conflict between synthesized eyes.

    Args:
        source_rgb: mono RGB source image (integer or float).
        left_rgb, right_rgb: synthesized RGB eyes with equal dimensions.
        map_left, map_right: exact production output-to-source normalized-U maps.
        shape: warp shape metadata containing valid ``content_scale_x/y``.
        warp_mask: optional packed or per-eye forward-hole mask. Any bilinear footprint touching
            a renderer hole is excluded so disocclusion fill cannot double-count as phase rivalry.
        source_sample_transform: optional RGB display transform applied *after* each fractional
            source sample, both on the canonical reference and the exact expected-eye paths.
            This preserves production order for a linear/scRGB source whose rendered eyes use a
            nonlinear HDR preview transform.
        max_analysis_width/height: canonical normalized-raster limits.  Evidence counts remain
            native-equivalent even when a smaller image is normalized onto this raster.
        min_phase_pixels: native-equivalent evidence floor.
        return_maps: return ``(metrics, maps)`` when visual evidence is needed.
    """
    prepared = prepare_interocular_evidence(
        source_rgb, left_rgb, right_rgb, map_left, map_right, shape,
        warp_mask=warp_mask, source_sample_transform=source_sample_transform,
        max_analysis_width=max_analysis_width,
        max_analysis_height=max_analysis_height)
    return measure_interocular_phase_chroma_prepared(
        prepared, min_phase_pixels=min_phase_pixels, return_maps=return_maps)


def measure_interocular_phase_chroma_prepared(
        prepared, *, min_phase_pixels=128, return_maps=False):
    """Measure phase/orientation conflict from shared registered evidence."""
    prepared = _require_prepared_evidence(prepared)
    reference = prepared.reference_rgb
    registered_left_raw = prepared.registered_left_rgb
    registered_right_raw = prepared.registered_right_rgb
    expected_left_registered = prepared.expected_left_rgb
    expected_right_registered = prepared.expected_right_rgb
    common = prepared.common_support
    evidence_basis = prepared.evidence_basis
    analysis_width = prepared.analysis_width

    # Remove global per-channel photometric changes before subtracting the expected sampling
    # residual.  Reversing that order would turn a harmless gain into
    # (gain - 1) * expected_residual, producing false phase around fractional edges.
    normalized_left_raw = _normalize_rgb_to_source(
        expected_left_registered, registered_left_raw, common)
    normalized_right_raw = _normalize_rgb_to_source(
        expected_right_registered, registered_right_raw, common)
    registered_left = reference + normalized_left_raw - expected_left_registered
    registered_right = reference + normalized_right_raw - expected_right_registered

    metrics = {
        "interocular_phase_orientation_burden_pct": None,
        "interocular_phase_orientation_support_pct": 0.0,
        "interocular_phase_orientation_support_count": 0,
        "interocular_phase_orientation_evidence_sufficient": 0.0,
    }

    reference_luma = reference @ _LUMA
    reference_common = reference_luma[common]
    source_span = (float(np.quantile(reference_common, 0.95)
                         - np.quantile(reference_common, 0.05))
                   if reference_common.size else 0.0)
    fill_left = np.where(common[..., None], registered_left, reference)
    fill_right = np.where(common[..., None], registered_right, reference)

    normalized_left = _normalize_rgb_to_source(reference, fill_left, common)
    normalized_right = _normalize_rgb_to_source(reference, fill_right, common)
    left_luma = normalized_left @ _LUMA
    right_luma = normalized_right @ _LUMA

    spatial_scale = analysis_width / 256.0
    base_radius = max(1, int(round(spatial_scale)))
    smoothing_radii = sorted(set((0, base_radius, 2 * base_radius)))
    phase_map = np.full(common.shape, np.nan, dtype=np.float32)
    phase_weight = np.zeros(common.shape, dtype=np.float32)
    phase_support = np.zeros(common.shape, dtype=bool)
    texture_floor = max(source_span * 0.010, 1.0 / 255.0)

    # Establish detail compatibility independently of the phase responses.  If one eye is
    # globally blurred, surviving low-frequency gradients can still have a different direction
    # around mixed structures; calling that phase rivalry would duplicate the dedicated detail
    # metric.  Local incompatibility is removed from support, and a frame dominated by it
    # explicitly abstains even if a small smooth subset remains measurable.
    balance_radius = max(1, int(round(analysis_width / 128.0)))
    balance_neighborhood = _erode_support(common, balance_radius + 1)
    source_detail = _registration._detail_energy(
        reference_luma, balance_radius) * spatial_scale
    left_detail = _registration._detail_energy(left_luma, balance_radius) * spatial_scale
    right_detail = _registration._detail_energy(right_luma, balance_radius) * spatial_scale
    detail_texture = balance_neighborhood & (source_detail >= texture_floor)
    detail_ratio = np.minimum(left_detail, right_detail) / np.maximum(
        np.maximum(left_detail, right_detail), 1e-8)
    detail_balance = detail_texture & (detail_ratio >= 0.70)
    detail_texture_count = int(np.count_nonzero(detail_texture))
    detail_balance_fraction = (
        np.count_nonzero(detail_balance) / detail_texture_count
        if detail_texture_count else 0.0)
    axis_radius = max(2, int(round(analysis_width / 64.0)))
    axis_source_detail = _registration._detail_energy(
        reference_luma, axis_radius) * spatial_scale
    axis_left_detail = _registration._detail_energy(left_luma, axis_radius) * spatial_scale
    axis_right_detail = _registration._detail_energy(right_luma, axis_radius) * spatial_scale
    axis_texture_floor = max(source_span * 0.0125, 1.5 / 255.0)
    axis_texture = balance_neighborhood & (axis_source_detail >= axis_texture_floor)
    axis_floor = max(axis_texture_floor * 0.20, 1e-5)
    detail_asymmetry = (
        200.0 * np.abs(axis_left_detail - axis_right_detail)
        / (axis_left_detail + axis_right_detail + 2.0 * axis_floor))
    detail_asymmetry_p90 = (
        float(np.quantile(detail_asymmetry[axis_texture], 0.90))
        if np.any(axis_texture) else np.inf)
    directional_imbalance = axis_texture & (detail_asymmetry >= 20.0)
    directional_count = int(np.count_nonzero(directional_imbalance))
    axis_texture_count = int(np.count_nonzero(axis_texture))
    if directional_count:
        left_stronger = int(np.count_nonzero(
            directional_imbalance & (axis_left_detail > axis_right_detail)))
        direction_dominance = max(
            left_stronger, directional_count - left_stronger) / directional_count
    else:
        direction_dominance = 0.0
    directional_fraction = directional_count / max(axis_texture_count, 1)
    global_detail_imbalance = (
        detail_asymmetry_p90 > 40.0 and directional_fraction >= 0.12
        and direction_dominance >= 0.80)
    frame_detail_compatible = (
        detail_texture_count > 0 and detail_balance_fraction >= 0.55
        and not global_detail_imbalance)

    for smoothing_radius in smoothing_radii:
        energy_radius = max(1, smoothing_radius + base_radius)
        margin = max(2, 2 * smoothing_radius + energy_radius + 1)
        neighborhood = _erode_support(common, margin)
        _, source_energy = _phase_responses(
            reference_luma, smoothing_radius, spatial_scale, energy_radius)
        left_response, left_energy = _phase_responses(
            left_luma, smoothing_radius, spatial_scale, energy_radius)
        right_response, right_energy = _phase_responses(
            right_luma, smoothing_radius, spatial_scale, energy_radius)

        strongest_eye = np.maximum(left_energy, right_energy)
        weakest_eye = np.minimum(left_energy, right_energy)
        # Phase is only meaningful when the eyes carry comparable response amplitude.  This gate
        # keeps unilateral blur/detail loss on the existing detail-asymmetry axis instead of
        # relabelling it as phase conflict.
        equal_detail = weakest_eye / np.maximum(strongest_eye, 1e-8) >= 0.60
        response_floor = max(texture_floor * 0.20, 1e-6)
        left_point_energy = np.linalg.norm(left_response, axis=2)
        right_point_energy = np.linalg.norm(right_response, axis=2)
        support = (neighborhood & detail_balance & (source_energy >= texture_floor)
                   & (weakest_eye >= 0.25 * texture_floor) & equal_detail
                   & (left_point_energy >= response_floor)
                   & (right_point_energy >= response_floor))
        if not np.any(support):
            continue

        # The source response does not enter the conflict itself; source energy selects and
        # weights real structure.  Keeping the response vectors source-independent avoids
        # rewarding two eyes merely because both are wrong in the same way (a separate
        # source-fidelity concern).
        cosine = np.sum(left_response * right_response, axis=2) / np.maximum(
            left_point_energy * right_point_energy, response_floor * response_floor)
        # Angular response distance is invariant to amplitude, unlike Euclidean unit-response
        # subtraction with local RMS denominators.  Symmetric low-pass blur retains phase and
        # orientation; a sign reversal reaches 100 and an orthogonal response reaches ~70.7.
        conflict = np.sqrt(np.maximum(0.5 * (1.0 - np.clip(cosine, -1.0, 1.0)), 0.0)) * 100.0
        capped_weight = np.minimum(
            source_energy, np.quantile(source_energy[support], 0.95))
        replace = support & (~phase_support | (conflict > phase_map))
        phase_map[replace] = conflict[replace]
        phase_weight[replace] = capped_weight[replace]
        phase_support |= support

    phase_count = _native_equivalent_count(phase_support, evidence_basis)
    metrics["interocular_phase_orientation_support_pct"] = float(
        100.0 * np.count_nonzero(phase_support) / max(phase_support.size, 1))
    metrics["interocular_phase_orientation_support_count"] = phase_count
    required_phase = max(
        int(min_phase_pixels), int(np.ceil(evidence_basis * 0.002)))
    if (phase_count >= required_phase and source_span >= 4.0 / 255.0
            and frame_detail_compatible):
        metrics["interocular_phase_orientation_evidence_sufficient"] = 100.0
        metrics["interocular_phase_orientation_burden_pct"] = _localized_burden(
            phase_map, phase_weight, phase_support, visibility_floor=5.0)

    if not return_maps:
        return metrics
    return metrics, {
        "registered_source_rgb": reference,
        "registered_left_rgb": registered_left_raw,
        "registered_right_rgb": registered_right_raw,
        "analysis_left_rgb": registered_left,
        "analysis_right_rgb": registered_right,
        "expected_left_rgb": expected_left_registered,
        "expected_right_rgb": expected_right_registered,
        "common_support": common,
        "phase_orientation_support": phase_support,
        "phase_detail_balance_support": detail_balance,
        "phase_directional_detail_imbalance": directional_imbalance,
        "phase_orientation_conflict_pct": phase_map,
    }


__all__ = [
    "PreparedInterocularEvidence",
    "prepare_interocular_evidence",
    "measure_interocular_phase_chroma",
    "measure_interocular_phase_chroma_prepared",
]
