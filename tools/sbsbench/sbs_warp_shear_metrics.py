"""Exact-map detector for unsupported horizontal row shear in one or two SBS eyes.

The ordinary warp Jacobian checks how source U advances from left to right.  It therefore misses
an important failure mode: adjacent output rows can request materially different horizontal
source positions while every row remains monotonic on its own.  On long, nearly horizontal
contours this appears as a serrated tear or a stack of displaced scanlines.

This module measures that cross-row derivative from the production inverse-warp map.  It works in
normalized image coordinates, reports the worst eye, and deliberately excludes:

* letterbox/pillarbox bars from the exact aspect-fit contract;
* source coordinates clamped by the live sampler;
* horizontally folded or severely collapsed mapping neighborhoods; and
* row changes coincident with a real horizontal boundary in the unwarped source image.

The last exclusion is important: a depth/disparity step at a source-supported horizontal material
boundary is legitimate geometry, not a renderer tear.  ``source`` is the original source image.
Callers that already reproduced production preprocessing may instead pass ``unwarped_source``: an
aspect-fitted, *unwarped* eye-sized image.  Do not pass an image sampled through ``source_u_map``;
that would let the suspect mapping manufacture its own supporting edge.

Only NumPy is required.  The implementation is deterministic and intentionally independent of
the evaluator/report plumbing so it can be corruption-qualified before becoming a gate or label.
"""

from __future__ import annotations

import math

import numpy as np


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)
_REFERENCE_EYE_WIDTH = 854.0
_REFERENCE_EYE_HEIGHT = 480.0


def _as_luma(image, name):
    value = np.asarray(image)
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] < 3):
        raise ValueError(f"{name} must be HxW or HxWx3+, got {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    if value.ndim == 3:
        value = value[..., :3] @ _LUMA
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _sample_uv(image, u, v):
    """Bilinearly sample a scalar image at normalized texture coordinates."""
    height, width = image.shape
    x = np.clip(u * width - 0.5, 0.0, width - 1.0)
    y = np.clip(v * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _dilate_rect(mask, radius_y, radius_x):
    """Small deterministic rectangular dilation without an optional SciPy dependency."""
    if radius_y <= 0 and radius_x <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, ((radius_y, radius_y), (radius_x, radius_x)), mode="constant")
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius_y + 1):
        for dx in range(2 * radius_x + 1):
            out |= padded[dy:dy + height, dx:dx + width]
    return out


def _validate_geometry(source_u_map, shape):
    mapping = np.asarray(source_u_map, dtype=np.float32)
    if mapping.ndim != 2:
        raise ValueError(f"source_u_map must be scalar HxW or Hx(2W), got {mapping.shape}")

    height = int(shape.get("eye_height", 0))
    eye_width = int(shape.get("eye_width", 0))
    source_width = int(shape.get("source_width", 0))
    source_height = int(shape.get("source_height", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if min(height, eye_width, source_width, source_height) <= 0:
        raise ValueError("mapping shape is missing positive eye/source dimensions")
    if not (0.0 < scale_x <= 1.0 and 0.0 < scale_y <= 1.0):
        raise ValueError("mapping shape has invalid content scale")
    if mapping.shape[0] != height or mapping.shape[1] not in (eye_width, 2 * eye_width):
        raise ValueError(
            f"source_u_map {mapping.shape} disagrees with eye geometry {(height, eye_width)}")
    if not np.isfinite(mapping).all():
        raise ValueError("source_u_map contains non-finite coordinates")
    if shape.get("height", height) != height:
        raise ValueError("packed mapping height disagrees with eye_height")
    if shape.get("width", 2 * eye_width) != 2 * eye_width:
        raise ValueError("packed mapping width disagrees with two-eye geometry")
    return mapping, height, eye_width, source_width, source_height, scale_x, scale_y


def _content_geometry(height, width, scale_x, scale_y):
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    source_u = (output_u - lo_x) / scale_x
    source_v = (output_v - lo_y) / scale_y
    content = ((output_u[None, :] >= lo_x) &
               (output_u[None, :] <= lo_x + scale_x) &
               (output_v[:, None] >= lo_y) &
               (output_v[:, None] <= lo_y + scale_y))
    return source_u, source_v, content


def _unwarped_evidence(source, unwarped_source, shape, source_u, source_v, content):
    height, width = content.shape
    if (source is None) == (unwarped_source is None):
        raise ValueError("supply exactly one of source or unwarped_source")
    if unwarped_source is not None:
        evidence = _as_luma(unwarped_source, "unwarped_source")
        if evidence.shape != (height, width):
            raise ValueError(
                f"unwarped_source {evidence.shape} != eye geometry {(height, width)}")
        return evidence

    source = _as_luma(source, "source")
    expected = (int(shape["source_height"]), int(shape["source_width"]))
    if source.shape != expected:
        raise ValueError(f"source {source.shape} != mapping source geometry {expected}")
    u = np.broadcast_to(np.clip(source_u, 0.0, 1.0)[None, :], (height, width))
    v = np.broadcast_to(np.clip(source_v, 0.0, 1.0)[:, None], (height, width))
    return _sample_uv(source, u, v)


def _source_boundary_strength(evidence, content):
    """Exposure-normalized vertical source gradients in the unwarped eye raster."""
    values = evidence[content]
    if values.size:
        lo, hi = np.percentile(values, (1.0, 99.0))
        dynamic_range = max(float(hi - lo), 1e-6)
    else:
        dynamic_range = 1e-6
    vertical_change = np.zeros_like(evidence, dtype=np.float32)
    vertical_change[1:] = np.abs(evidence[1:] - evidence[:-1])
    vertical_change[~content] = 0.0
    return vertical_change / dynamic_range


def _mapped_source_boundary(strength, mapping, scale_x, edge_fraction, dilation_ref_px):
    """Move native vertical-edge evidence horizontally through the suspect exact map.

    Sampling a precomputed vertical-gradient raster, rather than differentiating the mapped
    source image, is essential. The latter would convert an unsupported row shift across a
    vertical texture edge into evidence that falsely justifies that same shift.
    """
    height, width = mapping.shape
    lo_x = 0.5 * (1.0 - scale_x)
    sample_u = lo_x + np.clip(mapping, 0.0, 1.0) * scale_x
    sample_v = np.broadcast_to(
        ((np.arange(height, dtype=np.float32) + 0.5) / height)[:, None],
        mapping.shape)
    boundary = _sample_uv(strength, sample_u, sample_v) >= edge_fraction
    radius_y = max(1, int(round(dilation_ref_px * strength.shape[0] /
                                _REFERENCE_EYE_HEIGHT)))
    radius_x = max(1, int(round(dilation_ref_px * strength.shape[1] /
                                _REFERENCE_EYE_WIDTH)))
    return _dilate_rect(boundary, radius_y, radius_x)


def _horizontal_runs(mask, shear, threshold, width):
    burdens = []
    largest = 0
    for y, row in enumerate(mask):
        transitions = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(transitions == 1)
        ends = np.flatnonzero(transitions == -1)
        for start, end in zip(starts, ends):
            length = int(end - start)
            largest = max(largest, length)
            mean_ratio = float(np.mean(shear[y, start:end]) / threshold)
            burdens.append(length / float(width) * 100.0 * mean_ratio)
    return burdens, largest


def _measure_eye(mapping, source_boundary_strength, source_u, content, *, source_width,
                 scale_x, threshold, source_edge_fraction, source_edge_dilation_ref_px,
                 topology_min_step_ratio, min_support_count):
    height, width = mapping.shape
    finite = np.isfinite(mapping)
    live_u = np.clip(mapping, 0.0, 1.0)
    unclamped = (mapping > 0.0) & (mapping < 1.0)

    source_x = live_u * float(source_width)
    source_step = np.diff(source_x, axis=1)
    baseline_step = float(source_width) / max(scale_x * width, 1.0)
    healthy_step = source_step > topology_min_step_ratio * baseline_step
    healthy_topology = np.ones(mapping.shape, dtype=bool)
    healthy_topology[:, :-1] &= healthy_step
    healthy_topology[:, 1:] &= healthy_step

    source_boundary = _mapped_source_boundary(
        source_boundary_strength, mapping, scale_x, source_edge_fraction,
        source_edge_dilation_ref_px)

    pixel_support = content & finite & unclamped & healthy_topology
    pair_support = np.zeros_like(pixel_support)
    pair_support[1:] = pixel_support[1:] & pixel_support[:-1]
    pair_support &= ~source_boundary

    displacement = ((mapping - source_u[None, :]) * scale_x * width)
    row_shear = np.zeros_like(mapping, dtype=np.float32)
    # d(displacement / width) / d(y / height), expressed as px/row at the
    # 854x480 reference geometry. This is invariant under proportional resizing.
    normalization = (height / float(width)) * (
        _REFERENCE_EYE_WIDTH / _REFERENCE_EYE_HEIGHT)
    row_shear[1:] = np.abs(displacement[1:] - displacement[:-1]) * normalization

    bad = pair_support & (row_shear > threshold)
    support_count = int(np.count_nonzero(pair_support))
    content_pairs = np.zeros_like(content)
    content_pairs[1:] = content[1:] & content[:-1]
    support_pct = float(support_count / max(np.count_nonzero(content_pairs), 1) * 100.0)
    maps = {
        "row_shear_ref_px_per_row": row_shear,
        "support": pair_support,
        "bad": bad,
        "source_boundary": source_boundary,
        "healthy_topology": healthy_topology,
    }
    metrics = {
        "warp_cross_row_shear_support_pct": support_pct,
        "warp_cross_row_shear_support_count": support_count,
    }
    if support_count < min_support_count:
        metrics.update({
            "warp_cross_row_shear_severity_pct": None,
            "warp_cross_row_shear_bad_area_pct": None,
            "warp_cross_row_shear_largest_run_pct": None,
        })
        return metrics, maps

    burdens, largest = _horizontal_runs(bad, row_shear, threshold, width)
    strongest = sorted(burdens, reverse=True)[:4]
    severity = math.sqrt(float(np.mean(np.square(strongest)))) if strongest else 0.0
    metrics.update({
        "warp_cross_row_shear_severity_pct": severity,
        "warp_cross_row_shear_bad_area_pct": float(
            np.count_nonzero(bad) / support_count * 100.0),
        "warp_cross_row_shear_largest_run_pct": float(largest / width * 100.0),
    })
    return metrics, maps


def measure_cross_row_shear(source_u_map, mapping_shape, *, source=None,
                            unwarped_source=None, shear_threshold_ref_px=0.5,
                            source_edge_fraction=0.02, source_edge_dilation_ref_px=1.0,
                            topology_min_step_ratio=0.35, min_support_count=64,
                            return_maps=False):
    """Measure unsupported row-wise horizontal shear from an exact inverse-warp map.

    ``source_u_map`` may be one HxW eye or the harness' packed Hx(2W) map.  Packed input returns
    worst-eye severity/bad-area/run length and minimum per-eye support.  Spatial maps preserve the
    input packing when ``return_maps`` is true.

    ``shear_threshold_ref_px`` is pixels of horizontal displacement change per adjacent row at an
    854x480 eye.  The implementation converts it through normalized image coordinates, so a
    proportionally identical defect receives the same score at 480p, 1080p, or 2160p.

    Result keys:

    * ``warp_cross_row_shear_severity_pct``: RMS of the four strongest horizontal-run burdens;
      each burden is run-width percent times mean shear / threshold.
    * ``warp_cross_row_shear_bad_area_pct``: over-threshold pixels / qualified support.
    * ``warp_cross_row_shear_largest_run_pct``: longest bad run as percent of eye width.
    * ``warp_cross_row_shear_support_pct`` and ``_count``: minimum support in either input eye.

    Severity fields are ``None`` when any supplied eye has insufficient evidence.
    """
    if shear_threshold_ref_px <= 0.0:
        raise ValueError("shear_threshold_ref_px must be positive")
    if not (0.0 < source_edge_fraction < 1.0):
        raise ValueError("source_edge_fraction must be between zero and one")
    if source_edge_dilation_ref_px < 0.0:
        raise ValueError("source_edge_dilation_ref_px cannot be negative")
    if not (0.0 < topology_min_step_ratio < 1.0):
        raise ValueError("topology_min_step_ratio must be between zero and one")
    if min_support_count < 1:
        raise ValueError("min_support_count must be positive")

    (mapping, height, eye_width, source_width, _, scale_x,
     scale_y) = _validate_geometry(source_u_map, mapping_shape)
    source_u, source_v, content = _content_geometry(
        height, eye_width, scale_x, scale_y)
    evidence = _unwarped_evidence(
        source, unwarped_source, mapping_shape, source_u, source_v, content)
    source_boundary_strength = _source_boundary_strength(evidence, content)

    eye_count = mapping.shape[1] // eye_width
    per_eye = []
    per_eye_maps = []
    for eye_index in range(eye_count):
        eye_map = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        metrics, maps = _measure_eye(
            eye_map, source_boundary_strength, source_u, content, source_width=source_width,
            scale_x=scale_x, threshold=shear_threshold_ref_px,
            source_edge_fraction=source_edge_fraction,
            source_edge_dilation_ref_px=source_edge_dilation_ref_px,
            topology_min_step_ratio=topology_min_step_ratio,
            min_support_count=min_support_count)
        per_eye.append(metrics)
        per_eye_maps.append(maps)

    out = {
        "warp_cross_row_shear_support_pct": min(
            eye["warp_cross_row_shear_support_pct"] for eye in per_eye),
        "warp_cross_row_shear_support_count": min(
            eye["warp_cross_row_shear_support_count"] for eye in per_eye),
    }
    scored_keys = (
        "warp_cross_row_shear_severity_pct",
        "warp_cross_row_shear_bad_area_pct",
        "warp_cross_row_shear_largest_run_pct",
    )
    for key in scored_keys:
        values = [eye[key] for eye in per_eye]
        out[key] = None if any(value is None for value in values) else max(values)

    if not return_maps:
        return out
    packed_maps = {
        key: np.concatenate([eye[key] for eye in per_eye_maps], axis=1)
        for key in per_eye_maps[0]
    }
    return out, packed_maps
