"""Experimental exact-map metric for stereoscopic-window violations.

A stereo-window violation occurs when a surface with crossed (in-front-of-screen) disparity is
cut by a lateral image border.  The border says that the surface is behind the physical window,
while binocular disparity says that it is in front.  A large crossed disparity in the middle of
the image is therefore *not* a window violation.

This module inverts the two production output-to-source U maps onto one canonical source raster.
It keeps only unique correspondences supported by both eyes, then finds signed-disparity
components that actually reach the left or right visibility boundary.  Risk is restricted to the
border band that would be cut in one eye; it is not spread over the whole foreground object.
Raw-U clamps, folds, map gaps, letter/pillarbox bars, and optional invalid forward-coverage pixels
are excluded rather than filled.

The perceptual weighting follows the *principles* of the Disney Research window-violation model:
contrast magnitude, spatial frequency, orientation, and disparity all affect visibility.  This is
not a reproduction of Disney's display-calibrated psychophysical LUT: Apollo does not know the
viewer distance, physical display size, or calibrated luminance.  Consequently every public key
is explicitly ``experimental_`` and must be corruption-qualified before it can become a gate or
training label.

Only NumPy is required.  The analysis raster is capped at a fixed source-space width, making the
metric deterministic and approximately invariant to proportional output resolution changes.
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    from . import sbs_interocular_metrics as _geometry
except ImportError:  # Direct execution from tools/sbsbench.
    import sbs_interocular_metrics as _geometry


_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _as_float_image(image, name):
    value = np.asarray(image)
    if value.ndim not in (2, 3) or (value.ndim == 3 and value.shape[2] < 3):
        raise ValueError(f"{name} must be HxW or HxWx3+, got {value.shape}")
    if np.issubdtype(value.dtype, np.integer):
        value = value.astype(np.float32) / float(np.iinfo(value.dtype).max)
    else:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _as_luma(image, name):
    value = _as_float_image(image, name)
    return (value[..., :3] @ _LUMA).astype(np.float32) if value.ndim == 3 else value


def _box_mean(image, radius):
    image = np.asarray(image, dtype=np.float32)
    if radius <= 0:
        return image.copy()
    padded = np.pad(image, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    diameter = 2 * radius + 1
    total = (integral[diameter:, diameter:] - integral[:-diameter, diameter:]
             - integral[diameter:, :-diameter] + integral[:-diameter, :-diameter])
    return (total / float(diameter * diameter)).astype(np.float32)


def _sample_grid(image, width, height):
    """Bilinearly sample a scalar/RGB source at canonical pixel-center coordinates."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in (2, 3):
        raise ValueError(f"grid source must be HxW or HxWxC, got {image.shape}")
    source_height, source_width = image.shape[:2]
    u = (np.arange(width, dtype=np.float32) + 0.5) / float(width)
    v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    x = np.clip(u * source_width - 0.5, 0.0, source_width - 1.0)
    y = np.clip(v * source_height - 0.5, 0.0, source_height - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source_width - 1)
    y1 = np.minimum(y0 + 1, source_height - 1)
    fx, fy = x - x0, y - y0
    sampled_x0_y0 = image[y0[:, None], x0[None, :]]
    sampled_x1_y0 = image[y0[:, None], x1[None, :]]
    sampled_x0_y1 = image[y1[:, None], x0[None, :]]
    sampled_x1_y1 = image[y1[:, None], x1[None, :]]
    if image.ndim == 3:
        fx = fx[None, :, None]
        fy = fy[:, None, None]
    else:
        fx = fx[None, :]
        fy = fy[:, None]
    top = (1.0 - fx) * sampled_x0_y0 + fx * sampled_x1_y0
    bottom = (1.0 - fx) * sampled_x0_y1 + fx * sampled_x1_y1
    return ((1.0 - fy) * top + fy * bottom).astype(np.float32)


def _monotonic_runs(source_u, usable):
    """Yield increasing runs without interpolating through occlusion jumps or folds."""
    source_u = np.asarray(source_u, dtype=np.float32)
    indices = np.flatnonzero(np.asarray(usable, dtype=bool) & np.isfinite(source_u))
    if indices.size < 2:
        return
    adjacent = indices[1:] == indices[:-1] + 1
    differences = source_u[indices[1:]] - source_u[indices[:-1]]
    ordinary = differences[adjacent & (differences > 1e-8)]
    typical_step = float(np.median(ordinary)) if ordinary.size else 1.0 / source_u.size
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


def _invert_row(output_x, source_u, usable, target_u):
    """Recover output positions for source samples and reject ambiguous inverses."""
    count = np.zeros(target_u.size, dtype=np.int32)
    value_sum = np.zeros(target_u.size, dtype=np.float64)
    minimum = np.full(target_u.size, np.inf, dtype=np.float64)
    maximum = np.full(target_u.size, -np.inf, dtype=np.float64)
    # The live sampler clamps raw U.  Such pixels no longer identify a unique source coordinate.
    invertible = np.asarray(usable, dtype=bool) & (source_u >= 0.0) & (source_u <= 1.0)
    for run in _monotonic_runs(source_u, invertible):
        run_u = source_u[run]
        selected = ((target_u >= max(float(run_u[0]), 0.0) - 1e-7)
                    & (target_u <= min(float(run_u[-1]), 1.0) + 1e-7))
        if not selected.any():
            continue
        positions = np.interp(target_u[selected], run_u, output_x[run])
        value_sum[selected] += positions
        count[selected] += 1
        minimum[selected] = np.minimum(minimum[selected], positions)
        maximum[selected] = np.maximum(maximum[selected], positions)
    unique = (count > 0) & ((maximum - minimum) <= 1.05)
    position = np.full(target_u.size, np.nan, dtype=np.float32)
    position[unique] = (value_sum[unique] / count[unique]).astype(np.float32)
    return position, unique


def _healthy_inverse(position, valid, nominal_step):
    """Require locally ordered inverse support without bridging a source-space hole."""
    valid = np.asarray(valid, dtype=bool) & np.isfinite(position)
    healthy = np.zeros(valid.shape, dtype=bool)
    if position.size < 2:
        return healthy
    steps = np.diff(position)
    adjacent = valid[:-1] & valid[1:] & (steps > 1e-6)
    ordinary = steps[adjacent]
    typical = float(np.median(ordinary)) if ordinary.size else float(nominal_step)
    maximum = max(4.0 * float(nominal_step), 8.0 * typical)
    adjacent &= steps <= maximum
    healthy[:-1] |= adjacent
    healthy[1:] |= adjacent
    return healthy


def _validate(mapping, shape, source, depth, coverage_mask):
    mapping = np.asarray(mapping, dtype=np.float32)
    if mapping.ndim != 2 or mapping.shape[1] % 2:
        raise ValueError(f"source_u_map must be packed scalar Hx(2W), got {mapping.shape}")
    if not np.isfinite(mapping).all():
        raise ValueError("source_u_map contains non-finite coordinates")
    height, packed_width = mapping.shape
    eye_width = packed_width // 2
    expected = {
        "width": packed_width,
        "height": height,
        "eye_width": eye_width,
        "eye_height": height,
    }
    mismatch = {key: (value, shape.get(key)) for key, value in expected.items()
                if shape.get(key) != value}
    if mismatch:
        raise ValueError(f"mapping disagrees with shape contract: {mismatch}")
    source_width = int(shape.get("source_width", 0))
    source_height = int(shape.get("source_height", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if min(source_width, source_height) <= 0:
        raise ValueError("shape contract is missing positive source dimensions")
    if not (0.0 < scale_x <= 1.0 and 0.0 < scale_y <= 1.0):
        raise ValueError("shape contract has invalid content scales")
    source_image = _as_float_image(source, "source")
    if source_image.shape[:2] != (source_height, source_width):
        raise ValueError(
            f"source {source_image.shape[:2]} != mapping source geometry "
            f"{(source_height, source_width)}")
    if depth is not None:
        depth = _as_luma(depth, "depth")
        if depth.shape != source_image.shape[:2]:
            raise ValueError(f"depth {depth.shape} != source {source_image.shape[:2]}")
    if coverage_mask is not None:
        coverage_mask = np.asarray(coverage_mask, dtype=bool)
        if coverage_mask.shape != mapping.shape:
            raise ValueError(
                f"coverage_mask {coverage_mask.shape} != mapping {mapping.shape}")
    return (mapping, height, eye_width, source_width, source_height, scale_x, scale_y,
            source_image, depth, coverage_mask)


def _canonical_geometry(mapping, height, eye_width, source_width, source_height,
                        scale_x, scale_y, coverage_mask, analysis_max_width):
    content_width = max(2, int(round(scale_x * eye_width)))
    content_height = max(2, int(round(scale_y * height)))
    analysis_width = max(16, min(int(analysis_max_width), source_width, content_width))
    analysis_height = max(8, min(
        source_height, content_height,
        int(round(analysis_width * source_height / float(source_width)))))
    target_u = (np.arange(analysis_width, dtype=np.float32) + 0.5) / analysis_width
    target_v = (np.arange(analysis_height, dtype=np.float32) + 0.5) / analysis_height

    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / eye_width
    content_x = ((output_u >= lo_x) & (output_u <= lo_x + scale_x))
    output_x = np.arange(eye_width, dtype=np.float32)
    nominal_step = scale_x * eye_width / float(analysis_width)

    positions = [np.full((analysis_height, analysis_width), np.nan, dtype=np.float32)
                 for _ in range(2)]
    valid = [np.zeros((analysis_height, analysis_width), dtype=bool) for _ in range(2)]
    for target_row, source_v in enumerate(target_v):
        output_y = (lo_y + source_v * scale_y) * height - 0.5
        row = int(np.clip(round(output_y), 0, height - 1))
        for eye_index in range(2):
            offset = eye_index * eye_width
            usable = content_x.copy()
            if coverage_mask is not None:
                usable &= coverage_mask[row, offset:offset + eye_width]
            position, unique = _invert_row(
                output_x, mapping[row, offset:offset + eye_width], usable, target_u)
            unique &= _healthy_inverse(position, unique, nominal_step)
            positions[eye_index][target_row, unique] = position[unique]
            valid[eye_index][target_row, unique] = True

    common = valid[0] & valid[1]
    disparity_px = positions[1] - positions[0]
    disparity_pct = _geometry.perceived_disparity_pct(
        disparity_px, eye_width, height)
    return {
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "left_position": positions[0],
        "right_position": positions[1],
        "common": common,
        "disparity_px": disparity_px,
        "disparity_pct": disparity_pct,
        "content_left": lo_x * eye_width - 0.5,
        "content_right": (lo_x + scale_x) * eye_width - 0.5,
        "content_width": scale_x * eye_width,
    }


def _perceptual_weight(luma):
    """Display-agnostic contrast/frequency/orientation approximation in [0, 1]."""
    height, width = luma.shape
    # A one-reference-pixel optical prefilter keeps Nyquist alternation/JPEG grain from receiving
    # more weight merely because its derivative is larger.  It is the picture-space analogue of
    # the high-frequency falloff in a contrast-sensitivity function.
    prefilter_radius = max(1, int(round(width / 512.0)))
    perceptual_luma = _box_mean(luma, prefilter_radius)
    # Log-spaced bands approximate the paper's band-limited decomposition.  The mid bands are
    # weighted most strongly; without physical display/viewing geometry these are picture-space,
    # not cycles-per-degree, weights.
    base_radii = (1, 2, 4, 8)
    # Very fine alternation can carry large pixel derivatives while being less stereoscopically
    # salient than the mid bands.  Keep its weight deliberately low; otherwise JPEG grain and
    # one-pixel texture dominate the Disney-inspired CSF-like response.
    band_weights = (0.10, 0.25, 1.0, 0.55)
    detail = np.zeros_like(luma, dtype=np.float32)
    normalizer = 0.0
    seen = set()
    for base_radius, band_weight in zip(base_radii, band_weights):
        radius = max(1, int(round(base_radius * width / 512.0)))
        if radius in seen:
            continue
        seen.add(radius)
        broad_radius = min(max(radius + 1, 2 * radius), max(2, min(height, width) // 4))
        band = np.abs(
            _box_mean(perceptual_luma, radius)
            - _box_mean(perceptual_luma, broad_radius))
        detail += float(band_weight) * band
        normalizer += float(band_weight)
    detail /= max(normalizer, 1e-6)

    # Absolute contrast is intentional.  Per-frame range normalization would make a 1% contrast
    # pattern as disturbing as a 100% pattern, contradicting the psychophysical result.
    contrast_weight = np.clip(detail / 0.08, 0.0, 1.0)
    xpad = np.pad(perceptual_luma, ((0, 0), (1, 1)), mode="edge")
    ypad = np.pad(perceptual_luma, ((1, 1), (0, 0)), mode="edge")
    gx = np.abs(0.5 * (xpad[:, 2:] - xpad[:, :-2]))
    gy = np.abs(0.5 * (ypad[2:, :] - ypad[:-2, :]))
    # Disney's narrow horizontal-orientation condition was about 30% less disturbing.  A
    # horizontal line has predominantly vertical gradient, hence the 0.70 factor on gy.
    orientation_weight = ((gx + 0.70 * gy) / np.maximum(gx + gy, 1e-8))
    orientation_weight[gx + gy < 1e-8] = 0.70
    return (contrast_weight * orientation_weight).astype(np.float32), {
        "contrast_weight": contrast_weight.astype(np.float32),
        "orientation_weight": orientation_weight.astype(np.float32),
        "band_limited_contrast": detail.astype(np.float32),
    }


def _flood_components(mask, disparity_pct, depth, jump_pct, depth_jump):
    """Return component labels while refusing to bridge a disparity/depth discontinuity."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int32)
    components = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if labels[start_y, start_x] >= 0:
            continue
        label = len(components)
        labels[start_y, start_x] = label
        queue = deque([(int(start_y), int(start_x))])
        members = []
        while queue:
            y, x = queue.popleft()
            members.append(y * width + x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                           (y - 1, x - 1), (y - 1, x + 1),
                           (y + 1, x - 1), (y + 1, x + 1)):
                if (not (0 <= ny < height and 0 <= nx < width) or not mask[ny, nx]
                        or labels[ny, nx] >= 0):
                    continue
                if abs(float(disparity_pct[ny, nx] - disparity_pct[y, x])) > jump_pct:
                    continue
                if depth is not None and abs(float(depth[ny, nx] - depth[y, x])) > depth_jump:
                    continue
                labels[ny, nx] = label
                queue.append((ny, nx))
        components.append(np.asarray(members, dtype=np.int64))
    return labels, components


def _border_risk(sign_mask, geometry, *, crossed, component_jump_pct, depth,
                 depth_jump, seed_margin_px):
    left = geometry["left_position"]
    right = geometry["right_position"]
    disparity_px = geometry["disparity_px"]
    disparity_pct = geometry["disparity_pct"]
    content_left = geometry["content_left"]
    content_right = geometry["content_right"]

    if crossed:
        # xR < xL: the right-eye projection is cut at the left border and the left-eye
        # projection at the right border.
        left_distance = right - content_left
        right_distance = content_right - left
    else:
        # xR > xL: retain the opposite-sign border contact as a separate diagnostic, never call
        # it a foreground window violation.
        left_distance = left - content_left
        right_distance = content_right - right

    magnitude = np.abs(disparity_px)
    left_band = sign_mask & (left_distance >= -1e-4) & (
        left_distance <= magnitude + seed_margin_px)
    right_band = sign_mask & (right_distance >= -1e-4) & (
        right_distance <= magnitude + seed_margin_px)
    left_seed = sign_mask & (left_distance <= seed_margin_px) & (left_distance >= -1e-4)
    right_seed = sign_mask & (right_distance <= seed_margin_px) & (right_distance >= -1e-4)

    labels, components = _flood_components(
        sign_mask, disparity_pct, depth, component_jump_pct, depth_jump)
    left_risk = np.zeros(sign_mask.shape, dtype=bool)
    right_risk = np.zeros(sign_mask.shape, dtype=bool)
    flat_left_seed = left_seed.ravel()
    flat_right_seed = right_seed.ravel()
    flat_left_band = left_band.ravel()
    flat_right_band = right_band.ravel()
    flat_left_risk = left_risk.ravel()
    flat_right_risk = right_risk.ravel()
    for members in components:
        if flat_left_seed[members].any():
            chosen = members[flat_left_band[members]]
            flat_left_risk[chosen] = True
        if flat_right_seed[members].any():
            chosen = members[flat_right_band[members]]
            flat_right_risk[chosen] = True

    risk = left_risk | right_risk
    proximity = np.zeros(sign_mask.shape, dtype=np.float32)
    denominator = magnitude + seed_margin_px
    left_proximity = np.clip(1.0 - left_distance / denominator, 0.0, 1.0)
    right_proximity = np.clip(1.0 - right_distance / denominator, 0.0, 1.0)
    proximity[left_risk] = left_proximity[left_risk]
    proximity[right_risk] = np.maximum(
        proximity[right_risk], right_proximity[right_risk])
    return risk, left_risk, right_risk, proximity, labels


def _largest_component_area(mask, disparity_pct, depth, jump_pct, depth_jump):
    _, components = _flood_components(mask, disparity_pct, depth, jump_pct, depth_jump)
    return max((component.size for component in components), default=0)


def _metric_triplet(risk, contribution, support_count, disparity_pct, depth,
                    component_jump_pct, depth_jump):
    if support_count <= 0:
        return None, None, None
    burden = float(np.sum(contribution) / support_count * 100.0)
    area = float(np.count_nonzero(risk) / support_count * 100.0)
    largest = _largest_component_area(
        risk, disparity_pct, depth, component_jump_pct, depth_jump)
    largest_pct = float(largest / support_count * 100.0)
    return burden, area, largest_pct


def measure_stereo_window_violation(
        source_u_map, mapping_shape, source, *, depth=None, coverage_mask=None,
        source_sample_transform=None,
        disparity_threshold_pct=0.10, disparity_full_scale_pct=3.0,
        component_jump_pct=0.35, depth_jump=0.08, analysis_max_width=512,
        min_support_count=128, return_maps=False):
    """Measure exact, border-connected signed stereo-window risk.

    Args:
        source_u_map: Harness packed Hx(2W) raw inverse-U map, before sampler clamping.
        mapping_shape: Exact source/content/output geometry contract.
        source: Original source image at ``source_width`` x ``source_height``.  For HDR this must
            be the same linear RGB evidence sampled by the evaluated warp.  When
            ``source_sample_transform`` is supplied, the transform is applied only *after* the
            fractional canonical sample, matching the production texture-sample/color order.
        depth: Optional source-resolution depth.  It only prevents component connectivity across
            a depth jump; it does not determine crossed/uncrossed sign.
        coverage_mask: Optional packed boolean Hx(2W), true only for valid forward coverage.
        source_sample_transform: Optional nonlinear RGB-to-RGB preview/display transform.  It
            requires RGB source evidence, must preserve shape, and is never applied before
            bilinear sampling.  This distinction is material for FP16 HDR highlights.
        disparity_threshold_pct: Ignore numerical/no-percept disparity below this percentage of
            fitted eye-content width.
        disparity_full_scale_pct: Disparity contribution saturates at this image-relative value.
        component_jump_pct: Do not connect neighboring surfaces across a larger disparity jump.
        depth_jump: When depth is supplied, do not connect across a larger normalized jump.
        analysis_max_width: Fixed source-space cap used for resolution robustness.
        min_support_count: Abstain when fewer mutually supported canonical samples remain.
        return_maps: Return ``(metrics, maps)`` for detector-localized visual evidence.

    Returns:
        Crossed (in-front) burden, area, and largest-component percentages; opposite-sign border
        contact as independent ``uncrossed`` diagnostics; and exact mutual-support count/percent.
        All quality keys intentionally begin with ``experimental_``.  Burden combines normalized
        disparity, border proximity, band-limited contrast, and orientation.  Area and component
        size remain unweighted geometry evidence.
    """
    if not (0.0 < disparity_threshold_pct < disparity_full_scale_pct):
        raise ValueError("disparity threshold must be positive and below full scale")
    if component_jump_pct <= 0.0 or depth_jump <= 0.0:
        raise ValueError("component jump limits must be positive")
    if analysis_max_width < 16 or min_support_count < 1:
        raise ValueError("analysis/support limits are too small")

    (mapping, height, eye_width, source_width, source_height, scale_x, scale_y,
     source_image, depth, coverage_mask) = _validate(
         source_u_map, mapping_shape, source, depth, coverage_mask)
    geometry = _canonical_geometry(
        mapping, height, eye_width, source_width, source_height, scale_x, scale_y,
        coverage_mask, analysis_max_width)
    analysis_width = geometry["analysis_width"]
    analysis_height = geometry["analysis_height"]
    sampled_source = _sample_grid(source_image, analysis_width, analysis_height)
    if source_sample_transform is None:
        luma = (sampled_source[..., :3] @ _LUMA).astype(np.float32) \
            if sampled_source.ndim == 3 else sampled_source
    else:
        if sampled_source.ndim != 3:
            raise ValueError("source_sample_transform requires RGB source evidence")
        transformed = np.asarray(
            source_sample_transform(sampled_source[..., :3]), dtype=np.float32)
        if transformed.shape != sampled_source[..., :3].shape:
            raise ValueError(
                "source_sample_transform must preserve sampled RGB shape")
        if not np.isfinite(transformed).all():
            raise ValueError("source_sample_transform returned non-finite RGB evidence")
        luma = (transformed @ _LUMA).astype(np.float32)
    depth_grid = (_sample_grid(depth, analysis_width, analysis_height)
                  if depth is not None else None)
    perceptual_weight, perceptual_maps = _perceptual_weight(luma)

    common = geometry["common"]
    disparity_pct = geometry["disparity_pct"]
    support_count = int(np.count_nonzero(common))
    possible_count = int(common.size)
    output_sample_step = geometry["content_width"] / float(analysis_width)
    seed_margin_px = max(1.5, 1.5 * output_sample_step)

    crossed = common & (disparity_pct <= -disparity_threshold_pct)
    uncrossed = common & (disparity_pct >= disparity_threshold_pct)
    crossed_risk, crossed_left, crossed_right, crossed_proximity, _ = _border_risk(
        crossed, geometry, crossed=True, component_jump_pct=component_jump_pct,
        depth=depth_grid, depth_jump=depth_jump, seed_margin_px=seed_margin_px)
    uncrossed_risk, uncrossed_left, uncrossed_right, uncrossed_proximity, _ = _border_risk(
        uncrossed, geometry, crossed=False, component_jump_pct=component_jump_pct,
        depth=depth_grid, depth_jump=depth_jump, seed_margin_px=seed_margin_px)

    # Invalid inverse locations carry NaN by design.  Zero them before arithmetic: IEEE
    # ``NaN * False`` remains NaN and would otherwise poison the frame aggregate.
    magnitude = np.where(common, np.abs(disparity_pct), 0.0)
    disparity_weight = np.clip(
        (magnitude - disparity_threshold_pct)
        / (disparity_full_scale_pct - disparity_threshold_pct), 0.0, 1.0)
    crossed_contribution = (perceptual_weight * disparity_weight
                            * crossed_proximity * crossed_risk)
    uncrossed_contribution = (perceptual_weight * disparity_weight
                              * uncrossed_proximity * uncrossed_risk)

    prefix = "experimental_stereo_window_"
    metrics = {
        prefix + "support_count": support_count,
        prefix + "support_pct": float(support_count / max(possible_count, 1) * 100.0),
    }
    scored_keys = (
        prefix + "crossed_burden_pct",
        prefix + "crossed_area_pct",
        prefix + "crossed_largest_component_pct",
        prefix + "uncrossed_burden_pct",
        prefix + "uncrossed_area_pct",
        prefix + "uncrossed_largest_component_pct",
    )
    if support_count < min_support_count:
        metrics.update({key: None for key in scored_keys})
    else:
        crossed_values = _metric_triplet(
            crossed_risk, crossed_contribution, support_count, disparity_pct, depth_grid,
            component_jump_pct, depth_jump)
        uncrossed_values = _metric_triplet(
            uncrossed_risk, uncrossed_contribution, support_count, disparity_pct, depth_grid,
            component_jump_pct, depth_jump)
        metrics.update(dict(zip(scored_keys[:3], crossed_values)))
        metrics.update(dict(zip(scored_keys[3:], uncrossed_values)))

    if not return_maps:
        return metrics
    maps = {
        "source_luma": luma,
        "support": common,
        "disparity_pct": disparity_pct,
        "perceptual_weight": perceptual_weight,
        "crossed_risk": crossed_risk,
        "crossed_left_cut": crossed_left,
        "crossed_right_cut": crossed_right,
        "crossed_contribution": crossed_contribution,
        "uncrossed_risk": uncrossed_risk,
        "uncrossed_left_cut": uncrossed_left,
        "uncrossed_right_cut": uncrossed_right,
        "uncrossed_contribution": uncrossed_contribution,
        **perceptual_maps,
    }
    return metrics, maps


__all__ = ["measure_stereo_window_violation"]
