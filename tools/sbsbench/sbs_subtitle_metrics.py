"""Exact-map metrics for explicitly annotated burned-in subtitle regions.

The region mask is authenticated ground truth in source-frame coordinates.  This module does not
detect text and deliberately treats the mask as a loose region: callers may encode several
disjoint rectangles, and background enclosed by those rectangles remains part of the score.
"""

from __future__ import annotations

import numpy as np

import sbs_interocular_metrics


MIN_SUBTITLE_REGION_SAMPLES = 16
MIN_SOURCE_GRADIENT_ENERGY = (1.0 / 255.0) ** 2
MIN_SUBTITLE_TARGET_DISPARITY_PCT = -3.5
MAX_SUBTITLE_TARGET_DISPARITY_PCT = 3.0


def validate_subtitle_target_disparity_pct(value):
    """Return a finite signed xR-xL target inside the current comfort envelope."""
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or
            not np.issubdtype(np.asarray(value).dtype, np.number) or
            not np.isfinite(value)):
        raise ValueError("subtitle_target_disparity_pct must be a finite number")
    target = float(value)
    if not MIN_SUBTITLE_TARGET_DISPARITY_PCT <= target <= MAX_SUBTITLE_TARGET_DISPARITY_PCT:
        raise ValueError(
            "subtitle_target_disparity_pct must be within "
            f"[{MIN_SUBTITLE_TARGET_DISPARITY_PCT}, "
            f"{MAX_SUBTITLE_TARGET_DISPARITY_PCT}]")
    return target


def _sample_scalar_uv(image, u, v):
    """Bilinear normalized sampling with the D3D pixel-center convention."""
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape
    x = np.clip(np.asarray(u, dtype=np.float32) * width - 0.5, 0.0, width - 1.0)
    y = np.clip(np.asarray(v, dtype=np.float32) * height - 0.5, 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    return ((1.0 - wx) * (1.0 - wy) * image[y0, x0]
            + wx * (1.0 - wy) * image[y0, x1]
            + (1.0 - wx) * wy * image[y1, x0]
            + wx * wy * image[y1, x1]).astype(np.float32)


def _sample_region_nearest(region, u, v):
    """Sample an authored binary region without inventing feathered detector confidence."""
    region = np.asarray(region, dtype=bool)
    height, width = region.shape
    x = np.clip(np.floor(np.asarray(u, dtype=np.float32) * width).astype(np.int32),
                0, width - 1)
    y = np.clip(np.floor(np.asarray(v, dtype=np.float32) * height).astype(np.int32),
                0, height - 1)
    return region[y, x]


def _validate_geometry(binocular_geometry, shape):
    target_u = np.asarray(binocular_geometry.get("target_u"), dtype=np.float32)
    source_v = np.asarray(binocular_geometry.get("source_v"), dtype=np.float32)
    disparity = np.asarray(binocular_geometry.get("disparity"), dtype=np.float32)
    symmetry = np.asarray(binocular_geometry.get("symmetry"), dtype=np.float32)
    weight = np.asarray(binocular_geometry.get("weight"), dtype=np.float32)
    support_weight = np.asarray(
        binocular_geometry.get("support_weight"), dtype=np.float32)
    valid = np.asarray(binocular_geometry.get("valid"), dtype=bool)
    expected = (source_v.size, target_u.size)
    if (target_u.ndim != 1 or source_v.ndim != 1 or disparity.shape != expected or
            symmetry.shape != expected or weight.shape != expected or
            support_weight.shape != expected or valid.shape != expected):
        raise ValueError("subtitle metric binocular geometry is inconsistent")
    if (not np.isfinite(target_u).all() or not np.isfinite(source_v).all() or
            not np.isfinite(disparity[valid]).all() or
            not np.isfinite(symmetry[valid]).all() or
            not np.isfinite(weight[valid]).all() or
            not np.isfinite(support_weight).all()):
        raise ValueError("subtitle metric binocular geometry contains non-finite evidence")
    if np.any(weight[valid] <= 0.0) or np.any(support_weight < 0.0):
        raise ValueError("subtitle metric binocular geometry has non-positive valid weight")
    eye_width = int(shape.get("eye_width", 0))
    eye_height = int(shape.get("eye_height", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if (eye_width <= 0 or eye_height <= 0 or not 0.0 < scale_x <= 1.0 or
            not 0.0 < scale_y <= 1.0):
        raise ValueError("subtitle metric shape contract is incomplete")
    return target_u, source_v, disparity, symmetry, weight, support_weight, valid


def measure_subtitle_region(left, right, source, region, shape, binocular_geometry,
                            target_disparity_pct=None):
    """Measure zero-plane consistency and sharpness in explicit subtitle-region GT.

    Target-plane RMS error and disparity variance are output-area weighted and expressed in
    reference-aspect image-disparity percentage, avoiding resolution-dependent pixel units.
    Sharpness is the worse-eye ratio of registered post-warp to unwarped-source horizontal-gradient
    energy. Insufficient region or source-gradient evidence abstains instead of returning a false
    zero.
    """
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    region = np.asarray(region, dtype=bool)
    eye_height = int(shape.get("eye_height", 0))
    eye_width = int(shape.get("eye_width", 0))
    source_height = int(shape.get("source_height", 0))
    source_width = int(shape.get("source_width", 0))
    if left.shape != (eye_height, eye_width) or right.shape != left.shape:
        raise ValueError("subtitle metric eyes do not match the exact-map shape contract")
    if source.shape != (source_height, source_width) or region.shape != source.shape:
        raise ValueError("subtitle region must match the source-frame geometry")
    if not (np.isfinite(left).all() and np.isfinite(right).all() and
            np.isfinite(source).all()):
        raise ValueError("subtitle metric image evidence contains non-finite values")

    (target_u, source_v, disparity, symmetry, weight, support_weight,
     valid) = _validate_geometry(binocular_geometry, shape)
    source_u_grid = np.broadcast_to(target_u[None, :], valid.shape)
    source_v_grid = np.broadcast_to(source_v[:, None], valid.shape)
    sampled_region = _sample_region_nearest(region, source_u_grid, source_v_grid)
    authored_count = int(np.count_nonzero(sampled_region))
    supported = sampled_region & valid
    support_count = int(np.count_nonzero(supported))
    out = {
        "subtitle_region_authored_count": authored_count,
        "subtitle_region_support_count": support_count,
    }
    if authored_count:
        # Count coverage catches missing canonical correspondences. Output-area coverage catches
        # the complementary collapse where every canonical source sample survives but is squeezed
        # into a narrow rendered strip. The hard metric is the conservative minimum of the two.
        # Invalid samples never become zero-plane geometry evidence.
        sample_coverage_pct = support_count / authored_count * 100.0
        nominal_output_step = (
            float(shape["content_scale_x"]) * eye_width / target_u.size)
        nominal_authored_area = authored_count * nominal_output_step
        output_area_coverage_pct = (
            float(np.sum(support_weight[supported], dtype=np.float64)) /
            nominal_authored_area * 100.0)
        out["subtitle_region_binocular_support_pct"] = float(np.clip(
            min(sample_coverage_pct, output_area_coverage_pct), 0.0, 100.0))
    if support_count < MIN_SUBTITLE_REGION_SAMPLES:
        return out

    one_pixel_pct = sbs_interocular_metrics.perceived_disparity_pct(
        1.0, eye_width, eye_height)
    disparity_pct = disparity[supported].astype(np.float64) * one_pixel_pct
    area_weight = weight[supported].astype(np.float64)
    total_weight = float(np.sum(area_weight))
    weighted_mean = float(np.sum(disparity_pct * area_weight) / total_weight)
    out["subtitle_disparity_variance_pct2"] = float(
        np.sum(area_weight * (disparity_pct - weighted_mean) ** 2) / total_weight)
    if target_disparity_pct is not None:
        target = validate_subtitle_target_disparity_pct(target_disparity_pct)
        out["subtitle_target_disparity_rms_error_pct"] = float(np.sqrt(
            np.sum(area_weight * (disparity_pct - target) ** 2) / total_weight))

    # Invert the stored mean-camera residual and binocular disparity to obtain each eye's output
    # position for every canonical source sample.  Sampling the rendered eyes there registers the
    # post-warp text back onto the same grid used by the unwarped source denominator.
    scale_x = float(shape["content_scale_x"])
    scale_y = float(shape["content_scale_y"])
    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    unwarped_x = (lo_x + target_u * scale_x) * eye_width - 0.5
    mean_x = unwarped_x[None, :] + symmetry
    left_x = mean_x - 0.5 * disparity
    right_x = mean_x + 0.5 * disparity
    # Rejected folds/gaps carry NaN positions by contract. They do not vote, but the dense image
    # sampler still needs harmless finite coordinates for those array cells.
    left_x = np.where(valid, left_x, unwarped_x[None, :])
    right_x = np.where(valid, right_x, unwarped_x[None, :])
    output_v = np.broadcast_to(
        (lo_y + source_v * scale_y)[:, None], valid.shape)
    before = _sample_scalar_uv(source, source_u_grid, source_v_grid)
    after_left = _sample_scalar_uv(
        left, (left_x + 0.5) / float(eye_width), output_v)
    after_right = _sample_scalar_uv(
        right, (right_x + 0.5) / float(eye_width), output_v)

    gradient_support = supported[:, 1:] & supported[:, :-1]
    gradient_count = int(np.count_nonzero(gradient_support))
    if gradient_count < MIN_SUBTITLE_REGION_SAMPLES:
        return out
    before_gradient = np.diff(before, axis=1)[gradient_support].astype(np.float64)
    before_energy = float(np.mean(before_gradient * before_gradient))
    if before_energy < MIN_SOURCE_GRADIENT_ENERGY:
        return out
    ratios = []
    for after in (after_left, after_right):
        gradient = np.diff(after, axis=1)[gradient_support].astype(np.float64)
        ratios.append(float(np.mean(gradient * gradient) / before_energy))
    out["subtitle_sharpness_preservation_pct"] = float(min(ratios) * 100.0)
    return out


__all__ = [
    "MAX_SUBTITLE_TARGET_DISPARITY_PCT",
    "MIN_SUBTITLE_REGION_SAMPLES",
    "MIN_SUBTITLE_TARGET_DISPARITY_PCT",
    "measure_subtitle_region",
    "validate_subtitle_target_disparity_pct",
]
