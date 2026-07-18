"""Shared exact source-registration helpers for offline SBS metrics.

These routines invert an eye's production output-to-source U map onto a canonical source raster
while rejecting folds, clamped samples, disjoint source intervals, and ambiguous correspondences.
The phase/orientation and photometric-rivalry detectors plus exact binocular geometry use this
common implementation so their validity masks cannot drift apart.

This module deliberately exposes no quality score.  Its former low-frequency luma and local
detail-energy asymmetry scores failed authenticated-real benign controls and were removed rather
than retained as dormant model-label candidates.
"""

from __future__ import annotations

import numpy as np


REFERENCE_STREAM_ASPECT = 5120.0 / 2160.0


def perceived_disparity_pct(disparity_px, eye_width, eye_height):
    """Reference-aspect disparity normalization shared by every SBS geometry metric."""
    width = max(float(eye_width), 1.0)
    height = max(float(eye_height), 1.0)
    aspect_scale = (width / height) / REFERENCE_STREAM_ASPECT
    result = np.asarray(disparity_px, dtype=np.float32) * (100.0 / width * aspect_scale)
    return float(result) if result.ndim == 0 else result


def _box_mean(image, radius):
    """Reflect-padded square mean with an output shape identical to ``image``."""
    image = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return image.copy()
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    total = (integral[diameter:, diameter:] - integral[:-diameter, diameter:]
             - integral[diameter:, :-diameter] + integral[:-diameter, :-diameter])
    return total / float(diameter * diameter)


def _detail_energy(image, radius=2):
    """Local RMS central-gradient energy, robust to sub-pixel resampling phase."""
    image = np.asarray(image, dtype=np.float32)
    xpad = np.pad(image, ((0, 0), (1, 1)), mode="edge")
    ypad = np.pad(image, ((1, 1), (0, 0)), mode="edge")
    gx = 0.5 * (xpad[:, 2:] - xpad[:, :-2])
    gy = 0.5 * (ypad[2:, :] - ypad[:-2, :])
    return np.sqrt(np.maximum(_box_mean(gx * gx + gy * gy, radius), 0.0))


def _weighted_percentile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not keep.any():
        return None
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(values[min(np.searchsorted(cumulative, target, side="left"), values.size - 1)])


def _sample_source_grid(source, width, height):
    """Bilinearly sample the mono source at canonical pixel-center UV coordinates."""
    source = np.asarray(source, dtype=np.float32)
    source_height, source_width = source.shape
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    x = np.clip(u * source_width - 0.5, 0.0, source_width - 1.0)
    y = np.clip(v * source_height - 0.5, 0.0, source_height - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    fx, fy = x - x0, y - y0
    top = (1.0 - fx[None, :]) * source[y0[:, None], x0[None, :]]
    top += fx[None, :] * source[y0[:, None], x1[None, :]]
    bottom = (1.0 - fx[None, :]) * source[y1[:, None], x0[None, :]]
    bottom += fx[None, :] * source[y1[:, None], x1[None, :]]
    return ((1.0 - fy[:, None]) * top + fy[:, None] * bottom).astype(np.float32)


def _sample_output_row(image, output_y):
    height = image.shape[0]
    output_y = float(np.clip(output_y, 0.0, height - 1.0))
    y0 = int(np.floor(output_y))
    y1 = min(y0 + 1, height - 1)
    fraction = output_y - y0
    return ((1.0 - fraction) * image[y0] + fraction * image[y1]).astype(np.float32)


def _monotonic_runs(source_u, content):
    """Yield locally increasing contiguous inverse-map runs.

    Separate runs may cover the same source U after a fold.  The registration step retains their
    output positions and rejects such ambiguous overlap instead of averaging two surfaces.
    """
    source_u = np.asarray(source_u, dtype=np.float32)
    content = np.asarray(content, dtype=bool)
    indices = np.flatnonzero(content & np.isfinite(source_u))
    if indices.size < 2:
        return
    adjacent = indices[1:] == indices[:-1] + 1
    differences = source_u[indices[1:]] - source_u[indices[:-1]]
    ordinary = differences[adjacent & (differences > 1e-8)]
    typical_step = float(np.median(ordinary)) if ordinary.size else 1.0 / source_u.size
    # A large positive map jump is an occluded source interval, not evidence that every source
    # coordinate between its endpoints was sampled.  Splitting it is essential: np.interp across
    # that jump would manufacture mutual visibility.  Four output samples and an 8x robust local
    # allowance retain legitimate smooth stretch while rejecting discontinuities/fold seams.
    maximum_step = max(4.0 / source_u.size, 8.0 * typical_step)
    start = 0
    for offset in range(1, indices.size):
        previous, current = indices[offset - 1], indices[offset]
        step = float(source_u[current] - source_u[previous])
        if current != previous + 1 or step <= 1e-8 or step > maximum_step:
            run = indices[start:offset]
            if run.size >= 2:
                yield run
            start = offset
    run = indices[start:]
    if run.size >= 2:
        yield run


def _invert_row(eye_row, source_u_row, content_x, target_u):
    """Invert one exact source-U row while rejecting folded/multiple correspondences."""
    width = target_u.size
    value_sum = np.zeros(width, dtype=np.float64)
    count = np.zeros(width, dtype=np.int32)
    min_position = np.full(width, np.inf, dtype=np.float64)
    max_position = np.full(width, -np.inf, dtype=np.float64)
    # The live shader clamps raw U before sampling.  Samples outside [0, 1] therefore no longer
    # preserve an invertible source coordinate and must not be called mutually visible evidence.
    invertible = content_x & (source_u_row >= 0.0) & (source_u_row <= 1.0)
    for run in _monotonic_runs(source_u_row, invertible):
        run_u = source_u_row[run]
        lo, hi = max(float(run_u[0]), 0.0), min(float(run_u[-1]), 1.0)
        if hi < lo:
            continue
        selected = (target_u >= lo - 1e-7) & (target_u <= hi + 1e-7)
        if not selected.any():
            continue
        targets = target_u[selected]
        values = np.interp(targets, run_u, eye_row[run])
        positions = np.interp(targets, run_u, run.astype(np.float32))
        value_sum[selected] += values
        count[selected] += 1
        min_position[selected] = np.minimum(min_position[selected], positions)
        max_position[selected] = np.maximum(max_position[selected], positions)

    # Multiple adjacent runs may share their endpoint.  They represent the same sample when their
    # recovered positions differ by at most one output pixel; distant duplicates are a fold.
    unique = (count > 0) & ((max_position - min_position) <= 1.05)
    registered = np.full(width, np.nan, dtype=np.float32)
    registered[unique] = (value_sum[unique] / count[unique]).astype(np.float32)
    return registered, unique


def _robust_affine(source, eye, support):
    """Positive robust source->eye affine fit used only to isolate detail from exposure."""
    x = np.asarray(source[support], dtype=np.float64)
    y = np.asarray(eye[support], dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if x.size < 32 or float(np.std(x)) < 1e-6:
        return 1.0, 0.0
    for _ in range(3):
        design = np.stack((x, np.ones_like(x)), axis=1)
        gain, bias = np.linalg.lstsq(design, y, rcond=None)[0]
        gain = float(np.clip(gain, 0.25, 4.0))
        bias = float(np.median(y - gain * x))
        residual = np.abs(y - (gain * x + bias))
        cutoff = max(float(np.quantile(residual, 0.90)), 1e-6)
        inliers = residual <= cutoff
        if inliers.sum() < 32 or inliers.all():
            break
        x, y = x[inliers], y[inliers]
    return gain, bias


__all__ = []
