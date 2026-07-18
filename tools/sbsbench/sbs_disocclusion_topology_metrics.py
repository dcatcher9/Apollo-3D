"""Deterministic, source-relative disocclusion and foreground-leakage metrics.

The forward-coverage mask emitted by the harness is useful *geometry metadata*, but it is not a
quality verdict.  A renderer can fill every forward hole perfectly, or the mask itself can be
wrong.  This module therefore separates the validity classes used by stereo benchmarks and only
scores a visible defect when independent evidence agrees:

* ``non_occluded`` is in-content, in-frame output with forward coverage;
* ``disoccluded_raw`` is an in-frame forward-coverage hole before Apollo's fill;
* ``out_of_frame`` is content whose exact raw source U is outside [0, 1]; and
* ``disoccluded_supported`` is a raw hole connected to an independently measured depth edge.

The final eye is compared with the source sampled through the exact production inverse map.  A
supported hole only becomes ``bad_fill`` when that comparison contains a visible residual.
Foreground leakage/fattening is stricter: the residual must point toward the colour of the near
side of an independently measured depth edge.  An unrelated colour error can therefore vote for
bad fill, but not for foreground leakage.

This follows the fail-closed spirit of Middlebury/Sintel validity masks: occluded/disoccluded and
out-of-frame pixels are not silently folded into the mutually-visible denominator.  It does not
duplicate global source residual, stretch, fold, or clamp scores; all reported quality values are
localized to depth-supported topology opportunities.  NumPy is the only dependency.
"""

from __future__ import annotations

from collections import deque

import numpy as np


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _as_rgb(image, name):
    value = np.asarray(image)
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] < 3):
        raise ValueError(f"{name} must be HxW or HxWx3+, got {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    else:
        value = value[..., :3]
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _as_scalar(image, name):
    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] == 1:
        value = value[..., 0]
    if value.ndim != 2:
        raise ValueError(f"{name} must be HxW, got {value.shape}")
    value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _sample_uv(image, u, v):
    """Bilinearly sample HxW or HxWxC data at normalized texture coordinates."""
    image = np.asarray(image, dtype=np.float32)
    height, width = image.shape[:2]
    x = np.clip(u * width - 0.5, 0.0, width - 1.0)
    y = np.clip(v * height - 0.5, 0.0, height - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    fx, fy = x - x0, y - y0
    if image.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    top = image[y0, x0] * (1.0 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1.0 - fx) + image[y1, x1] * fx
    return (top * (1.0 - fy) + bottom * fy).astype(np.float32)


def _validate_shape(shape, mapping, left, right, source):
    eye_width = int(shape.get("eye_width", 0))
    eye_height = int(shape.get("eye_height", 0))
    source_width = int(shape.get("source_width", 0))
    source_height = int(shape.get("source_height", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if min(eye_width, eye_height, source_width, source_height) <= 0:
        raise ValueError("mapping shape is missing positive eye/source dimensions")
    if not (0.0 < scale_x <= 1.0 and 0.0 < scale_y <= 1.0):
        raise ValueError("mapping shape has invalid content scale")
    if shape.get("width", 2 * eye_width) != 2 * eye_width:
        raise ValueError("mapping shape packed width is not two eye widths")
    if shape.get("height", eye_height) != eye_height:
        raise ValueError("mapping shape height disagrees with eye_height")
    if mapping.shape != (eye_height, 2 * eye_width):
        raise ValueError(
            f"source_u_map {mapping.shape} != packed eye geometry "
            f"{(eye_height, 2 * eye_width)}")
    if left.shape[:2] != (eye_height, eye_width):
        raise ValueError(f"left eye {left.shape[:2]} != {(eye_height, eye_width)}")
    if right.shape[:2] != (eye_height, eye_width):
        raise ValueError(f"right eye {right.shape[:2]} != {(eye_height, eye_width)}")
    if source.shape[:2] != (source_height, source_width):
        raise ValueError(f"source {source.shape[:2]} != {(source_height, source_width)}")
    return eye_width, eye_height, scale_x, scale_y


def _content_coordinates(height, width, scale_x, scale_y):
    output_u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    content_x = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    content_y = (output_v >= lo_y) & (output_v <= lo_y + scale_y)
    content = content_y[:, None] & content_x[None, :]
    source_u = (output_u - lo_x) / scale_x
    source_v = (output_v - lo_y) / scale_y
    return content, source_u, source_v


def _mask_red(warp_mask, expected_shape):
    value = np.asarray(warp_mask)
    if value.ndim == 3:
        if value.shape[2] < 1:
            raise ValueError("warp_mask has no red channel")
        value = value[..., 0]
    if value.shape != expected_shape:
        raise ValueError(f"warp_mask {value.shape} != source_u_map {expected_shape}")
    if not np.isfinite(value).all():
        raise ValueError("warp_mask contains non-finite values")
    if np.issubdtype(value.dtype, np.integer):
        threshold = float(np.iinfo(value.dtype).max) * 0.5
    else:
        maximum = float(np.max(value)) if value.size else 0.0
        threshold = 127.5 if maximum > 1.5 else 0.5
    return value > threshold


def _normalize_depth(depth):
    depth = _as_scalar(depth, "depth")
    lo, hi = np.quantile(depth, (0.02, 0.98))
    span = float(hi - lo)
    if span <= 1e-7:
        return np.zeros_like(depth), False
    return np.clip((depth - lo) / span, 0.0, 1.0).astype(np.float32), True


def _connected_components(mask):
    """Return 4-connected flat-index arrays in stable raster order."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    for flat in np.flatnonzero(mask):
        y, x = divmod(int(flat), width)
        if visited[y, x]:
            continue
        visited[y, x] = True
        queue = deque(((y, x),))
        members = []
        while queue:
            cy, cx = queue.popleft()
            members.append(cy * width + cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                           (cy, cx - 1), (cy, cx + 1)):
                if (0 <= ny < height and 0 <= nx < width and mask[ny, nx]
                        and not visited[ny, nx]):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        components.append(np.asarray(members, dtype=np.int64))
    return components


def _dilate_horizontal(mask, radius):
    if radius <= 0:
        return np.asarray(mask, dtype=bool).copy()
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    padded = np.pad(mask, ((0, 0), (radius, radius)), mode="constant")
    out = np.zeros_like(mask)
    for offset in range(2 * radius + 1):
        out |= padded[:, offset:offset + width]
    return out


def _predicted_disocclusion_opportunities(
    mapping,
    identity_grid,
    content,
    boundary,
    foreground_valid,
    content_width,
    minimum_shift_jump_px=0.5,
):
    """Locate visible depth edges whose exact-map parallax changes materially.

    The forward-hole mask is renderer evidence, not self-authenticating ground truth.  If that
    mask is accidentally cleared, treating ``raw_hole_count == 0`` as a perfect fill would make
    the detector fail open.  A genuine disocclusion opportunity requires two independent facts:

    * a colour-distinguishable source-depth boundary (``foreground_valid``); and
    * a depth-dependent jump in the exact inverse-map displacement.

    A common camera translation has no displacement jump and therefore remains a legitimate
    no-hole case.  Off-source/clamped coordinates are excluded because they are frame-boundary
    topology, not an in-frame disocclusion.  The returned mask is expressed on pixels adjacent to
    the qualifying pair so it can be inspected in validator evidence.
    """
    mapping = np.asarray(mapping, dtype=np.float32)
    identity_grid = np.asarray(identity_grid, dtype=np.float32)
    content = np.asarray(content, dtype=bool)
    boundary = np.asarray(boundary, dtype=bool)
    foreground_valid = np.asarray(foreground_valid, dtype=bool)
    if not (mapping.shape == identity_grid.shape == content.shape == boundary.shape
            == foreground_valid.shape):
        raise ValueError("disocclusion opportunity inputs must share one geometry")
    if content_width <= 0.0 or minimum_shift_jump_px <= 0.0:
        raise ValueError("disocclusion opportunity scales must be positive")

    in_frame = np.isfinite(mapping) & (mapping >= 0.0) & (mapping <= 1.0)
    displacement_px = (mapping - identity_grid) * float(content_width)
    pair_support = (content[:, :-1] & content[:, 1:]
                    & in_frame[:, :-1] & in_frame[:, 1:])
    visible_edge_pair = ((boundary[:, :-1] | boundary[:, 1:])
                         & (foreground_valid[:, :-1] | foreground_valid[:, 1:]))
    shift_jump = np.abs(displacement_px[:, 1:] - displacement_px[:, :-1])
    pair_opportunity = (pair_support & visible_edge_pair
                        & (shift_jump >= float(minimum_shift_jump_px)))
    opportunity = np.zeros(mapping.shape, dtype=bool)
    opportunity[:, :-1] |= pair_opportunity
    opportunity[:, 1:] |= pair_opportunity
    return opportunity


def _largest_component_area_pct(mask, content_count):
    components = _connected_components(mask)
    largest = max((component.size for component in components), default=0)
    return float(largest / max(content_count, 1) * 100.0)


def _thin_depth_edges(delta, pair_valid, threshold):
    """Keep one peak per contiguous same-direction depth transition.

    Bilinear sampling spreads one source-depth step over multiple output pairs when the eye is
    larger than the depth texture.  Counting every pair would make topology support grow with
    resolution.  Stable peak selection restores one material boundary without changing its sign.
    """
    candidate = pair_valid & (np.abs(delta) >= threshold)
    selected = np.zeros(candidate.shape, dtype=bool)
    for y in range(candidate.shape[0]):
        indices = np.flatnonzero(candidate[y])
        if not indices.size:
            continue
        start = 0
        signs = np.sign(delta[y, indices])
        for offset in range(1, indices.size + 1):
            boundary = (offset == indices.size or indices[offset] != indices[offset - 1] + 1
                        or signs[offset] != signs[offset - 1])
            if not boundary:
                continue
            run = indices[start:offset]
            strengths = np.abs(delta[y, run])
            maximum = float(np.max(strengths))
            # Equal bilinear lobes are common. Choose their middle, not a resolution-dependent
            # first/last edge produced by floating-point tie noise.
            maxima = run[np.flatnonzero(strengths >= maximum - 1e-7)]
            selected[y, maxima[len(maxima) // 2]] = True
            start = offset
    return selected


def _depth_topology(mapped_depth, valid, reference, *, near_is_high, edge_threshold,
                    band_radius, prototype_offset, min_material_contrast, dynamic_range):
    """Find background-side topology opportunities and their foreground colour vectors."""
    height, width = mapped_depth.shape
    delta = mapped_depth[:, 1:] - mapped_depth[:, :-1]
    pair_valid = valid[:, 1:] & valid[:, :-1]
    edge = _thin_depth_edges(delta, pair_valid, edge_threshold)
    if not near_is_high:
        delta = -delta

    # Positive delta means left=background, right=foreground. Negative is the reverse.
    left_background = edge & (delta > 0.0)
    right_background = edge & (delta < 0.0)
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:, :-1] |= edge
    boundary[:, 1:] |= edge

    foreground = np.zeros_like(reference, dtype=np.float32)
    foreground_depth = np.zeros_like(mapped_depth, dtype=np.float32)
    foreground_valid = np.zeros((height, width), dtype=bool)
    foreground_distance = np.full((height, width), np.iinfo(np.int32).max, dtype=np.int32)

    # Assign each background-side sample the nearest foreground reference.  A normalized band
    # makes a proportional silhouette defect comparable across 480p, 1080p, and 2160p.
    for distance in range(band_radius):
        if distance + prototype_offset < width - 1:
            # Edge pair x/x+1, background extends left from x.
            targets = left_background[:, distance:width - 1 - prototype_offset]
            target_x = slice(0, width - 1 - distance - prototype_offset)
            foreground_x = slice(1 + distance + prototype_offset, width)
            material = np.sqrt(np.mean(
                (reference[:, foreground_x] - reference[:, target_x]) ** 2,
                axis=2)) / dynamic_range * 100.0
            select = (targets & (material >= min_material_contrast)
                      & (foreground_distance[:, target_x] > distance))
            foreground[:, target_x][select] = reference[:, foreground_x][select]
            foreground_depth[:, target_x][select] = mapped_depth[:, foreground_x][select]
            foreground_distance[:, target_x][select] = distance
            foreground_valid[:, target_x][select] = True

            # Edge pair x/x+1, background extends right from x+1.
            targets = right_background[:, prototype_offset:width - 1 - distance]
            target_x = slice(1 + distance + prototype_offset, width)
            foreground_x = slice(0, width - 1 - distance - prototype_offset)
            material = np.sqrt(np.mean(
                (reference[:, foreground_x] - reference[:, target_x]) ** 2,
                axis=2)) / dynamic_range * 100.0
            select = (targets & (material >= min_material_contrast)
                      & (foreground_distance[:, target_x] > distance))
            foreground[:, target_x][select] = reference[:, foreground_x][select]
            foreground_depth[:, target_x][select] = mapped_depth[:, foreground_x][select]
            foreground_distance[:, target_x][select] = distance
            foreground_valid[:, target_x][select] = True

    return (boundary, foreground_valid & valid, foreground, mapped_depth.copy(),
            foreground_depth)


def _propagate_edge_prototypes(valid, background, foreground, background_depth,
                               foreground_depth, radius):
    """Propagate the nearest independently observed edge pair to both sides of its boundary."""
    valid = np.asarray(valid, dtype=bool)
    height, width = valid.shape
    out_valid = np.zeros_like(valid)
    out_background = np.zeros_like(background)
    out_foreground = np.zeros_like(foreground)
    out_background_depth = np.zeros_like(background_depth)
    out_foreground_depth = np.zeros_like(foreground_depth)
    distance_map = np.full(valid.shape, np.iinfo(np.int32).max, dtype=np.int32)
    for offset in range(-radius, radius + 1):
        distance = abs(offset)
        if offset < 0:
            source_x = slice(-offset, width)
            target_x = slice(0, width + offset)
        elif offset > 0:
            source_x = slice(0, width - offset)
            target_x = slice(offset, width)
        else:
            source_x = target_x = slice(0, width)
        select = valid[:, source_x] & (distance_map[:, target_x] > distance)
        out_valid[:, target_x][select] = True
        distance_map[:, target_x][select] = distance
        out_background[:, target_x][select] = background[:, source_x][select]
        out_foreground[:, target_x][select] = foreground[:, source_x][select]
        out_background_depth[:, target_x][select] = background_depth[:, source_x][select]
        out_foreground_depth[:, target_x][select] = foreground_depth[:, source_x][select]
    return (out_valid, out_background, out_foreground, out_background_depth,
            out_foreground_depth)


def _measure_eye(eye, mapping, hole_mask, source, depth, content, identity_u, source_v, *,
                 source_sample_transform, near_is_high, depth_edge_threshold,
                 visibility_threshold_pct, material_contrast_threshold_pct,
                 foreground_projection_threshold, foreground_alignment_threshold,
                 topology_band_fraction, min_supported_hole_pixels,
                 min_foreground_support_pixels):
    height, width = mapping.shape
    in_frame = np.isfinite(mapping) & (mapping >= 0.0) & (mapping <= 1.0)
    out_of_frame = content & ~in_frame
    raw_hole = content & in_frame & hole_mask
    non_occluded = content & in_frame & ~hole_mask

    sample_u = np.clip(mapping, 0.0, 1.0)
    sample_v = np.broadcast_to(np.clip(source_v, 0.0, 1.0)[:, None], mapping.shape)
    reference = _sample_uv(source, sample_u, sample_v)
    identity_grid = np.broadcast_to(
        np.clip(identity_u, 0.0, 1.0)[None, :], mapping.shape)
    unwarped_reference = _sample_uv(source, identity_grid, sample_v)
    if source_sample_transform is not None:
        reference = np.asarray(source_sample_transform(reference), dtype=np.float32)
        unwarped_reference = np.asarray(
            source_sample_transform(unwarped_reference), dtype=np.float32)
        if (reference.shape != eye.shape or unwarped_reference.shape != eye.shape
                or not np.isfinite(reference).all()
                or not np.isfinite(unwarped_reference).all()):
            raise ValueError("source_sample_transform returned invalid RGB evidence")
    mapped_depth = _sample_uv(depth, sample_u, sample_v)
    unwarped_depth = _sample_uv(depth, identity_grid, sample_v)

    reference_luma = reference @ _LUMA
    eye_luma = eye @ _LUMA
    values = reference_luma[content & in_frame]
    if values.size:
        lo, hi = np.quantile(values, (0.01, 0.99))
        luma_range = float(hi - lo)
        rgb_ranges = [float(np.quantile(reference[..., channel][content & in_frame], 0.99)
                            - np.quantile(reference[..., channel][content & in_frame], 0.01))
                      for channel in range(3)]
        dynamic_range = max(luma_range, float(np.mean(rgb_ranges)), 4.0 / 255.0)
    else:
        dynamic_range = 4.0 / 255.0

    content_width = max(1, int(np.max(np.count_nonzero(content, axis=1))))
    band_radius = max(3, int(round(content_width * topology_band_fraction)))
    # Move far enough onto the foreground plateau that bilinear phase at the material boundary
    # cannot change the reference with output resolution.  This is expressed in content width,
    # not full-eye width, so pillarboxing does not alter the contract.
    prototype_offset = max(1, int(round(content_width * 0.00625)))
    (boundary, base_foreground_valid, base_foreground, base_background_depth,
     base_foreground_depth) = _depth_topology(
        unwarped_depth, content, unwarped_reference, near_is_high=near_is_high,
        edge_threshold=depth_edge_threshold, band_radius=band_radius,
        prototype_offset=prototype_offset,
        min_material_contrast=material_contrast_threshold_pct,
        dynamic_range=dynamic_range)
    (foreground_valid, independent_background, foreground,
     independent_background_depth, independent_foreground_depth) = (
        _propagate_edge_prototypes(
            base_foreground_valid, unwarped_reference, base_foreground,
            base_background_depth, base_foreground_depth,
            max(band_radius, int(round(content_width * 0.03)))))
    edge_near = foreground_valid
    predicted_opportunity = _predicted_disocclusion_opportunities(
        mapping, identity_grid, content, boundary, foreground_valid,
        content_width=float(content_width))

    # A forward-mask component is accepted only when a nontrivial part of it lies near an
    # independent depth discontinuity.  One accidental overlap cannot validate a full-frame mask.
    supported_hole = np.zeros_like(raw_hole)
    for component in _connected_components(raw_hole):
        near_count = int(np.count_nonzero(edge_near.ravel()[component]))
        required = max(1, int(np.ceil(component.size * 0.03)))
        if near_count >= required:
            supported_hole.ravel()[component] = True

    # Authenticate the exact map's chosen material against independent unwarped depth.  A map
    # landing on the background may retain its exact texture coordinate; a map landing on the
    # foreground inside a forward hole must not define its own "perfect" reference.
    map_background = (foreground_valid &
                      (np.abs(mapped_depth - independent_background_depth)
                       <= np.abs(mapped_depth - independent_foreground_depth)))
    use_independent_background = supported_hole & foreground_valid & ~map_background
    expected_background = reference.copy()
    expected_background[use_independent_background] = (
        independent_background[use_independent_background])

    expected_luma = expected_background @ _LUMA
    rgb_residual = np.sqrt(np.mean((eye - expected_background) ** 2, axis=2))
    luma_residual = np.abs(eye_luma - expected_luma)
    residual_pct = np.maximum(luma_residual, 0.75 * rgb_residual) / dynamic_range * 100.0
    bad_fill = supported_hole & (residual_pct >= visibility_threshold_pct)
    bad_fill_fraction = np.where(
        bad_fill, np.clip(residual_pct / 100.0, 0.0, 1.0), 0.0).astype(np.float32)
    leak_support = (foreground_valid &
                    (supported_hole | (non_occluded & map_background)))
    direction = foreground - expected_background
    observed = eye - expected_background
    direction_norm2 = np.sum(direction * direction, axis=2)
    observed_norm = np.sqrt(np.sum(observed * observed, axis=2))
    direction_norm = np.sqrt(direction_norm2)
    projection = np.divide(
        np.sum(observed * direction, axis=2), np.maximum(direction_norm2, 1e-12),
        out=np.zeros_like(mapped_depth), where=direction_norm2 > 1e-12)
    alignment = np.divide(
        np.sum(observed * direction, axis=2),
        np.maximum(observed_norm * direction_norm, 1e-12),
        out=np.zeros_like(mapped_depth),
        where=(observed_norm > 1e-8) & (direction_norm > 1e-8))
    foreground_leak = (leak_support & (residual_pct >= visibility_threshold_pct)
                       & (projection >= foreground_projection_threshold)
                       & (alignment >= foreground_alignment_threshold))
    leak_fraction = np.where(
        foreground_leak, np.clip(projection, 0.0, 1.0), 0.0).astype(np.float32)

    content_count = int(np.count_nonzero(content))
    raw_hole_count = int(np.count_nonzero(raw_hole))
    supported_hole_count = int(np.count_nonzero(supported_hole))
    leak_support_count = int(np.count_nonzero(leak_support))
    required_holes = max(
        int(min_supported_hole_pixels), int(np.ceil(content_count * 0.00005)))
    required_leak = max(
        int(min_foreground_support_pixels), int(np.ceil(content_count * 0.0001)))

    metrics = {
        "disocclusion_raw_hole_count": raw_hole_count,
        "disocclusion_predicted_opportunity_count": int(
            np.count_nonzero(predicted_opportunity)),
        "disocclusion_topology_support_count": supported_hole_count,
        "disocclusion_topology_support_pct": (
            100.0 * supported_hole_count / max(raw_hole_count, 1)),
        "foreground_leak_support_count": leak_support_count,
        "foreground_leak_support_pct": (
            100.0 * leak_support_count / max(content_count, 1)),
    }

    if raw_hole_count == 0 and np.any(predicted_opportunity):
        metrics.update({
            "disocclusion_bad_fill_evidence_sufficient": 0.0,
            "disocclusion_bad_fill_abstained": 1.0,
            "disocclusion_bad_fill_pct": None,
            "disocclusion_bad_fill_burden_pct": None,
            "disocclusion_bad_fill_area_pct": None,
            "disocclusion_bad_fill_largest_component_pct": None,
        })
    elif raw_hole_count == 0:
        metrics.update({
            "disocclusion_bad_fill_evidence_sufficient": 100.0,
            "disocclusion_bad_fill_abstained": 0.0,
            "disocclusion_bad_fill_pct": 0.0,
            "disocclusion_bad_fill_burden_pct": 0.0,
            "disocclusion_bad_fill_area_pct": 0.0,
            "disocclusion_bad_fill_largest_component_pct": 0.0,
        })
    elif supported_hole_count < required_holes:
        metrics.update({
            "disocclusion_bad_fill_evidence_sufficient": 0.0,
            "disocclusion_bad_fill_abstained": 1.0,
            "disocclusion_bad_fill_pct": None,
            "disocclusion_bad_fill_burden_pct": None,
            "disocclusion_bad_fill_area_pct": None,
            "disocclusion_bad_fill_largest_component_pct": None,
        })
    else:
        metrics.update({
            "disocclusion_bad_fill_evidence_sufficient": 100.0,
            "disocclusion_bad_fill_abstained": 0.0,
            "disocclusion_bad_fill_pct": float(
                np.count_nonzero(bad_fill) / supported_hole_count * 100.0),
            "disocclusion_bad_fill_burden_pct": float(
                np.sum(bad_fill_fraction) / max(content_count, 1) * 100.0),
            "disocclusion_bad_fill_area_pct": float(
                np.count_nonzero(bad_fill) / max(content_count, 1) * 100.0),
            "disocclusion_bad_fill_largest_component_pct": (
                _largest_component_area_pct(bad_fill, content_count)),
        })

    if leak_support_count < required_leak:
        metrics.update({
            "foreground_leak_evidence_sufficient": 0.0,
            "foreground_leak_abstained": 1.0,
            "foreground_leak_burden_pct": None,
            "foreground_leak_area_pct": None,
            "foreground_leak_largest_component_pct": None,
        })
    else:
        metrics.update({
            "foreground_leak_evidence_sufficient": 100.0,
            "foreground_leak_abstained": 0.0,
            "foreground_leak_burden_pct": float(
                np.sum(leak_fraction) / max(content_count, 1) * 100.0),
            "foreground_leak_area_pct": float(
                np.count_nonzero(foreground_leak) / max(content_count, 1) * 100.0),
            "foreground_leak_largest_component_pct": (
                _largest_component_area_pct(foreground_leak, content_count)),
        })

    maps = {
        "content": content,
        "non_occluded": non_occluded,
        "disoccluded_raw": raw_hole,
        "disoccluded_supported": supported_hole,
        "out_of_frame": out_of_frame,
        "depth_topology_boundary": boundary,
        "predicted_disocclusion_opportunity": predicted_opportunity,
        "bad_fill": bad_fill,
        "foreground_leak_support": leak_support,
        "foreground_leak": foreground_leak,
        "source_residual_pct": residual_pct.astype(np.float32),
        "foreground_projection": projection.astype(np.float32),
        "mapped_reference": reference,
        "expected_background_reference": expected_background,
        "foreground_reference": foreground,
        "independent_background_reference": independent_background,
        "map_background_authenticated": map_background,
    }
    return metrics, maps


def measure_disocclusion_topology(
        source, left, right, source_u_map, warp_mask, depth, mapping_shape, *,
        source_sample_transform=None, near_is_high=True, depth_edge_threshold=0.08,
        visibility_threshold_pct=3.0, material_contrast_threshold_pct=4.0,
        foreground_projection_threshold=0.15, foreground_alignment_threshold=0.75,
        topology_band_fraction=0.025, min_supported_hole_pixels=8,
        min_foreground_support_pixels=16, return_maps=False):
    """Measure visible bad fills and foreground leakage at depth-supported topology edges.

    Args:
        source: Original mono source RGB/luma in the same source geometry as ``mapping_shape``.
        left, right: Final synthesized eyes.  Integer images are normalized by their dtype.
        source_u_map: Packed Hx(2W) raw production output-to-source U map, before clamp.
        warp_mask: Packed mask whose red channel is the pre-fill forward disocclusion flag.
        depth: Source-space high-near depth by default.  Its resolution may differ from source.
        mapping_shape: Exact aspect-fit/source/eye geometry contract.
        source_sample_transform: Optional display transform applied *after* fractional source
            sampling.  This is required when source and final eye are in different transfer
            functions, for example linear HDR source versus a tonemapped preview eye.
        near_is_high: Set false only for an explicitly authenticated low-near depth contract.
        return_maps: Return ``(metrics, packed_maps)`` when true.

    ``None`` means insufficient independent topology evidence, never a perfect zero.  Clean scenes
    with no raw holes report zero bad-fill burden, while foreground leakage still abstains if the
    scene contains no colour-distinguishable depth edge.  Packed map values are worst-eye metrics;
    support is the minimum per-eye evidence so one unsupported eye cannot hide behind the other.
    """
    if not (0.0 < depth_edge_threshold < 1.0):
        raise ValueError("depth_edge_threshold must be between zero and one")
    if visibility_threshold_pct <= 0.0 or material_contrast_threshold_pct <= 0.0:
        raise ValueError("visibility/material thresholds must be positive")
    if not (0.0 < foreground_projection_threshold <= 1.0):
        raise ValueError("foreground_projection_threshold must be in (0, 1]")
    if not (0.0 < foreground_alignment_threshold <= 1.0):
        raise ValueError("foreground_alignment_threshold must be in (0, 1]")
    if not (0.0 < topology_band_fraction <= 0.1):
        raise ValueError("topology_band_fraction must be in (0, 0.1]")
    if min_supported_hole_pixels < 1 or min_foreground_support_pixels < 1:
        raise ValueError("minimum support counts must be positive")

    source = _as_rgb(source, "source")
    left, right = _as_rgb(left, "left"), _as_rgb(right, "right")
    mapping = _as_scalar(source_u_map, "source_u_map")
    eye_width, eye_height, scale_x, scale_y = _validate_shape(
        mapping_shape, mapping, left, right, source)
    red = _mask_red(warp_mask, mapping.shape)
    depth, depth_has_range = _normalize_depth(depth)
    content, identity_u, source_v = _content_coordinates(
        eye_height, eye_width, scale_x, scale_y)

    per_eye = []
    per_eye_maps = []
    for eye_index, eye in enumerate((left, right)):
        eye_map = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        eye_mask = red[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        metrics, maps = _measure_eye(
            eye, eye_map, eye_mask, source, depth, content, identity_u, source_v,
            source_sample_transform=source_sample_transform,
            near_is_high=near_is_high,
            depth_edge_threshold=(depth_edge_threshold if depth_has_range else 2.0),
            visibility_threshold_pct=visibility_threshold_pct,
            material_contrast_threshold_pct=material_contrast_threshold_pct,
            foreground_projection_threshold=foreground_projection_threshold,
            foreground_alignment_threshold=foreground_alignment_threshold,
            topology_band_fraction=topology_band_fraction,
            min_supported_hole_pixels=min_supported_hole_pixels,
            min_foreground_support_pixels=min_foreground_support_pixels)
        per_eye.append(metrics)
        per_eye_maps.append(maps)

    minimum_keys = (
        "disocclusion_topology_support_count",
        "disocclusion_topology_support_pct",
        "foreground_leak_support_count",
        "foreground_leak_support_pct",
        "disocclusion_bad_fill_evidence_sufficient",
        "foreground_leak_evidence_sufficient",
    )
    maximum_keys = (
        "disocclusion_raw_hole_count",
        "disocclusion_predicted_opportunity_count",
        "disocclusion_bad_fill_abstained",
        "disocclusion_bad_fill_pct",
        "disocclusion_bad_fill_burden_pct",
        "disocclusion_bad_fill_area_pct",
        "disocclusion_bad_fill_largest_component_pct",
        "foreground_leak_abstained",
        "foreground_leak_burden_pct",
        "foreground_leak_area_pct",
        "foreground_leak_largest_component_pct",
    )
    metrics = {key: min(eye[key] for eye in per_eye) for key in minimum_keys}
    for key in maximum_keys:
        values = [eye[key] for eye in per_eye]
        metrics[key] = None if any(value is None for value in values) else max(values)

    if not return_maps:
        return metrics
    packed_maps = {}
    for key in per_eye_maps[0]:
        if per_eye_maps[0][key].ndim == 3:
            packed_maps[key] = np.concatenate(
                (per_eye_maps[0][key], per_eye_maps[1][key]), axis=1)
        else:
            packed_maps[key] = np.concatenate(
                (per_eye_maps[0][key], per_eye_maps[1][key]), axis=1)
    return metrics, packed_maps


__all__ = ["measure_disocclusion_topology"]
