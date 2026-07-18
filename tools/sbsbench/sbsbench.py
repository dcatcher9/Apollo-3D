#!/usr/bin/env python3
"""
sbsbench - validated visual metrics for Apollo's host SBS 3D output.

Runs on real "Dump 3D" output (the actual sbs.png the client receives + the depth.png
that produced it), so the numbers reflect the LIVE pipeline, not a CPU replica -- this is
the whole point vs. warpsim (see docs/sbs-benchmark-plan.md). Each metric is a number that
should move with a real quality change, so improvements can be A/B'd against a baseline.

Metric families:
  exact visible stereo Production inverse-map disparity, weighted by independent horizontal
                        source structure. This separates requested geometry from visible relief.
  vmisalign_p99         Localized exact-map |dy| between eyes. Should be ~0; nonzero = a geometry
                        fault (eyes must differ by horizontal parallax only).
  source coverage /     Hard integrity constraints after horizontally aligning each eye to the
  image integrity       source; catches missing, black, or collapsed output regions.
  warp topology         Exact-map stretch, folds, clamps, cross-row shear, and binocular geometry
                        defects. These do not infer artifacts from source texture alone.
  GT depth accuracy     Scale/shift-invariant RMSE and boundary F1 on clips with gt_depth sidecars.
  GT depth lag          Whether predicted boundaries match the previous GT frame better than the
                        current frame, directly detecting held-depth temporal registration.
  flow temporal         Output residual after authenticated optical-flow compensation.

Usage:
  python sbsbench.py DUMP_DIR [DUMP_DIR ...]        # one or more dump_* folders
  python sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_2026070*"   # shell-free globbing
  python sbsbench.py DUMP ... --json out.json       # write the scorecard
  python sbsbench.py DUMP ... --baseline base.json  # print deltas vs a saved scorecard

Dependencies: numpy + Pillow only.
"""
import argparse
import atexit
import concurrent.futures
import glob
import json
import os
import re
import sys

# Scoring is parallel across frames. A BLAS team inside every worker oversubscribes the CPU, so
# spawned workers inherit a single-threaded numeric runtime and the process pool owns parallelism.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402  (thread limits must precede numeric-runtime import)
from PIL import Image  # noqa: E402

import sbs_interocular_metrics  # noqa: E402
import sbs_interocular_phase_chroma  # noqa: E402
import sbs_interocular_photometric_rivalry  # noqa: E402
import sbs_stereo_window_metrics  # noqa: E402
import sbs_warp_shear_metrics  # noqa: E402

TEMPORAL_MIN_SUPPORT = 0.1
VERTICAL_MISALIGNMENT_QUANTILE = 0.99
# Minimum independently measured support for a detector to become applicable.  A positive count
# is not automatically enough. Percentage/fraction supports use a strict positive threshold because
# their producing metric already enforces its own statistical minimum.
EVIDENCE_SUPPORT_REQUIREMENTS = {
    "source_fidelity_support_pct": 1e-12,
    "image_integrity_support": 0.1,
    "exact_mapping_support_pct": 1e-12,
    "exact_binocular_support_count": 1024.0,
    "exact_polarity_support_pct": 1e-12,
    "exact_local_polarity_support_pct": 1e-12,
    "exact_visible_support_pct": 1e-12,
    "exact_local_polarity_support_count": 256.0,
    "exact_visible_support_count": 256.0,
    "vmisalign_support_pct": 2.0,
    "warp_cross_row_shear_support_count": 512.0,
    "experimental_stereo_window_support_count": 128.0,
    "interocular_phase_orientation_evidence_sufficient": 100.0,
    "interocular_exposure_rivalry_evidence_sufficient": 100.0,
    "interocular_color_gain_rivalry_evidence_sufficient": 100.0,
}


# ---------------------------------------------------------------------------- io

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def rgb_luma(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def load_gray(path):
    # Rec.709 luma; the depth PNG is grayscale so any channel would do, but this is general.
    return rgb_luma(load_rgb(path))


def load_depth(path):
    """Depth map -> float array. NPY preserves public-dataset metric depth; image sidecars and
    harness depth retain their existing normalized representation."""
    if path.lower().endswith(".npy"):
        depth = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).squeeze()
        if depth.ndim != 2:
            raise ValueError(f"GT depth must be HxW, got {depth.shape}: {path}")
        return depth
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I"):
        return np.asarray(im, dtype=np.float32) / 65535.0
    if im.mode == "L":
        return np.asarray(im, dtype=np.float32) / 255.0
    return load_gray(path)


def load_warp_mapping(path, shape):
    """Load the harness-only exact inverse-warp map under its fail-closed shape contract."""
    required = {
        "schema": 1,
        "dtype": "float32-le",
        "layout": "row-major",
        "channels": ["raw_reproject_source_u_normalized"],
    }
    mismatched = {key: (expected, shape.get(key)) for key, expected in required.items()
                  if shape.get(key) != expected}
    width, height = shape.get("width"), shape.get("height")
    eye_width, eye_height = shape.get("eye_width"), shape.get("eye_height")
    source_width, source_height = shape.get("source_width"), shape.get("source_height")
    scale_x, scale_y = shape.get("content_scale_x"), shape.get("content_scale_y")
    valid_geometry = (
        isinstance(width, int) and width > 0 and width % 2 == 0
        and isinstance(height, int) and height > 0
        and isinstance(eye_width, int) and eye_width * 2 == width
        and eye_height == height
        and isinstance(source_width, int) and source_width > 0
        and isinstance(source_height, int) and source_height > 0
        and isinstance(scale_x, (int, float)) and 0.0 < scale_x <= 1.0
        and isinstance(scale_y, (int, float)) and 0.0 < scale_y <= 1.0)
    if mismatched or not valid_geometry:
        raise ValueError(f"invalid warp-map shape contract: fields={mismatched}, shape={shape}")
    values = np.fromfile(path, dtype="<f4")
    expected_values = width * height
    if values.size != expected_values:
        raise ValueError(
            f"warp-map size mismatch: {values.size} floats != {expected_values}: {path}")
    mapping = values.reshape(height, width)
    if not np.isfinite(mapping).all():
        raise ValueError(f"warp map contains non-finite values: {path}")
    return mapping


def split_eyes(sbs_gray):
    """SBS is [left | right], each half the width."""
    if sbs_gray.ndim < 2 or sbs_gray.shape[1] % 2:
        raise ValueError(f"packed SBS width must be even, got shape {sbs_gray.shape}")
    w = sbs_gray.shape[1] // 2
    return sbs_gray[:, :w], sbs_gray[:, w:2 * w]


def indexed_files(pattern, prefix):
    """Return {numeric_frame_id: path}, rejecting ambiguous names/duplicates.

    Eval assets are joined by identity, never list position. Positional zipping silently compared
    frame N against N+1 whenever extraction or rendering dropped a frame.
    """
    out = {}
    rx = re.compile(r"^" + re.escape(prefix) + r"(\d+)")
    for path in glob.glob(pattern):
        m = rx.match(os.path.basename(path))
        if not m:
            continue
        frame_id = int(m.group(1))
        if frame_id in out:
            raise ValueError(f"duplicate {prefix} frame id {frame_id}: {out[frame_id]} and {path}")
        out[frame_id] = path
    return out


def predecessor_frame_id(indexed, frame_id):
    """Previous available numeric identity, independent of padding or contiguity."""
    return max((candidate for candidate in indexed if candidate < frame_id), default=None)


# ------------------------------------------------ image registration / exact stereo geometry

def phase_shift(a, b):
    """Sub-tile (dy, dx) that best aligns b onto a, via phase correlation with sub-pixel
    parabolic refinement around the peak (wraparound-safe). Signed, in pixels. Without the
    refinement pop quantizes to integers, which is coarse on small-resolution clips."""
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    mag[mag < 1e-8] = 1e-8
    corr = np.fft.irfft2(r / mag, s=a.shape)
    h, w = a.shape
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def refine(cm, c0, cp):
        d = cm - 2.0 * c0 + cp
        return 0.5 * (cm - cp) / d if abs(d) > 1e-12 else 0.0

    dy = py + refine(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = px + refine(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    return float(dy), float(dx)


def translation_residual(a, b, dy, dx):
    """Mean absolute error after an integer non-wrapping translation of b onto a."""
    iy, ix = int(round(dy)), int(round(dx))
    h, w = a.shape
    ay0, ay1 = max(0, iy), min(h, h + iy)
    ax0, ax1 = max(0, ix), min(w, w + ix)
    by0, by1 = max(0, -iy), min(h, h - iy)
    bx0, bx1 = max(0, -ix), min(w, w - ix)
    if ay0 >= ay1 or ax0 >= ax1:
        return float("inf")
    return float(np.mean(np.abs(a[ay0:ay1, ax0:ax1] - b[by0:by1, bx0:bx1])))


def disparity_field(left, right, tile=192, stride=128, min_var=1e-3):
    """Per-tile horizontal/vertical disparity between the eyes, weighted by tile texture.
    Only textured tiles (variance > min_var) vote, so flat sky/UI doesn't wash out the stats."""
    h, w = left.shape
    dxs, dys, wts = [], [], []
    tile = max(8, min(tile, h, w))
    stride = max(1, min(stride, tile))
    for y in _tile_positions(h, tile, stride):
        for x in _tile_positions(w, tile, stride):
            lt = left[y:y + tile, x:x + tile]
            rt = right[y:y + tile, x:x + tile]
            v = float(lt.var())
            if v < min_var:
                continue
            dy, dx = phase_shift(lt, rt)
            # A shift near the unambiguous range edge is unreliable; drop it.
            if abs(dx) >= tile // 2 - 1 or abs(dy) >= tile // 2 - 1:
                continue
            # Repetitive texture can create a strong but false phase-correlation peak. A real
            # multi-pixel eye displacement must improve non-wrapping photometric alignment; an
            # alias that is no better than the unshifted eyes must never drive a hard comfort gate.
            if max(abs(dx), abs(dy)) >= 2.0:
                aligned = translation_residual(lt, rt, dy, dx)
                unaligned = float(np.mean(np.abs(lt - rt)))
                if aligned >= unaligned * 0.9:
                    continue
            dxs.append(dx)
            dys.append(dy)
            wts.append(v)
    if not dxs:
        return None
    dxs = np.array(dxs)
    dys = np.array(dys)
    wts = np.array(wts)
    return dxs, dys, wts


def _exact_expected_source_luma(eye_shape, src_gray, sampled_source_u, shape,
                                src_rgb=None, hdr_scale=None):
    """Reconstruct the luma sampled by one production eye and its content mask.

    The warp-map sidecar records the horizontal source coordinate consumed by the shader.  The
    vertical coordinate is the deterministic content-fit transform.  Keeping this reconstruction
    in one helper prevents vertical-alignment and source-fidelity metrics from drifting apart.
    """
    sampled_source_u = np.asarray(sampled_source_u, dtype=np.float32)
    if sampled_source_u.shape != tuple(eye_shape):
        raise ValueError(
            f"source-coordinate eye {sampled_source_u.shape} != output eye {eye_shape}")
    if not np.isfinite(sampled_source_u).all():
        raise ValueError("source-coordinate eye contains non-finite coordinates")
    height, eye_width = eye_shape
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if not (0.0 < scale_x <= 1.0 and 0.0 < scale_y <= 1.0):
        raise ValueError("exact source metric requires valid content scales")

    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / float(eye_width)
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lo_x, lo_y = 0.5 * (1.0 - scale_x), 0.5 * (1.0 - scale_y)
    content = ((output_v[:, None] >= lo_y) & (output_v[:, None] <= lo_y + scale_y)
               & (output_u[None, :] >= lo_x) & (output_u[None, :] <= lo_x + scale_x))
    valid = content & np.isfinite(sampled_source_u)
    if not valid.any():
        raise ValueError("exact source metric has no valid content pixels")
    source_v = np.clip((output_v - lo_y) / scale_y, 0.0, 1.0)
    source_v = np.broadcast_to(source_v[:, None], sampled_source_u.shape)
    live_sampled_u = np.clip(sampled_source_u, 0.0, 1.0)

    expected_rgb = None
    if hdr_scale is not None and src_rgb is None:
        raise ValueError("HDR preview fidelity requires RGB source evidence")
    if src_rgb is None:
        expected = _sample_scalar_uv(src_gray, live_sampled_u, source_v)
    elif hdr_scale is None:
        expected_rgb = _sample_rgb_uv(src_rgb, live_sampled_u, source_v)
        expected = rgb_luma(expected_rgb)
    else:
        linear_source = (_srgb_to_linear(src_rgb) * float(hdr_scale)).astype(
            np.float16).astype(np.float32)
        expected_rgb = _hdr_preview_rgb(
            _sample_rgb_uv(linear_source, live_sampled_u, source_v))
        expected = rgb_luma(expected_rgb)
    return expected, expected_rgb, valid


def _local_vertical_offsets(eye, expected, valid, max_height=540,
                            min_std=3.0 / 255.0):
    """Estimate vertical source-relative displacement on overlapping normalized tiles.

    Zero-mean normalized correlation makes the estimate insensitive to local exposure/contrast.
    Candidate reference patches are taken from surrounding *rows*, rather than correlating two
    cropped tiles with wraparound, so one-pixel defects remain measurable.  Ambiguous/repetitive
    or low-texture patches abstain instead of voting a random displacement.
    """
    eye = np.asarray(eye, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    original_height, original_width = eye.shape
    scale = min(1.0, float(max_height) / max(original_height, 1))
    height = max(24, int(round(original_height * scale)))
    width = max(32, int(round(original_width * scale)))
    if (height, width) != eye.shape:
        eye = resize_to(eye, width, height)
        expected = resize_to(expected, width, height)
        valid = resize_mask_conservative(valid, width, height)

    tile = max(16, min(64, int(round(height * 0.10))))
    max_shift = max(2, min(12, int(np.ceil(height * 0.03))))
    if height < tile + 2 * max_shift:
        tile = max(8, height - 2 * max_shift)
    stride = max(4, tile // 2)
    y_positions = [y for y in _tile_positions(height, tile, stride)
                   if y >= max_shift and y + tile + max_shift <= height]
    x_positions = _tile_positions(width, tile, stride)
    if not y_positions:
        return {}, 0

    offsets = {}
    attempted = len(y_positions) * len(x_positions)
    min_pixels = max(32, int(round(tile * tile * 0.90)))
    candidate_offsets = range(-max_shift, max_shift + 1)
    for y in y_positions:
        for x in x_positions:
            output_valid = valid[y:y + tile, x:x + tile]
            if int(output_valid.sum()) < min_pixels:
                continue
            output = eye[y:y + tile, x:x + tile]
            output_values = output[output_valid]
            output_centered = output_values - float(output_values.mean())
            output_norm = float(np.linalg.norm(output_centered))
            output_std = output_norm / np.sqrt(max(output_centered.size, 1))
            if output_std < min_std:
                continue

            scores = []
            texture = []
            for shift in candidate_offsets:
                ry = y - shift
                reference_valid = valid[ry:ry + tile, x:x + tile]
                joint = output_valid & reference_valid
                if int(joint.sum()) < min_pixels:
                    scores.append(float("-inf"))
                    texture.append(0.0)
                    continue
                a = output[joint]
                b = expected[ry:ry + tile, x:x + tile][joint]
                a = a - float(a.mean())
                b = b - float(b.mean())
                na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                reference_std = nb / np.sqrt(max(b.size, 1))
                if min(na, nb) <= 1e-8 or reference_std < min_std:
                    scores.append(float("-inf"))
                    texture.append(0.0)
                    continue
                scores.append(float(np.dot(a, b) / (na * nb)))
                texture.append(reference_std)

            best_index = int(np.argmax(scores))
            best_score = scores[best_index]
            if not np.isfinite(best_score) or best_score < 0.55:
                continue
            # A repeated vertical pattern can have several equally valid peaks.  Do not turn its
            # arbitrary phase into a hard stereoscopic fault.
            separated = [score for index, score in enumerate(scores)
                         if abs(index - best_index) > 1 and np.isfinite(score)]
            if separated and best_score - max(separated) < 0.004:
                continue

            shift = float(best_index - max_shift)
            if 0 < best_index < len(scores) - 1:
                before, peak, after = scores[best_index - 1:best_index + 2]
                denominator = before - 2.0 * peak + after
                if (np.isfinite(before) and np.isfinite(after)
                        and abs(denominator) > 1e-8):
                    shift += float(0.5 * (before - after) / denominator)
            # Texture is capped so one high-contrast logo cannot outweigh the rest of a frame.
            weight = min(0.10, np.sqrt(max(output_std * texture[best_index], 0.0)))
            offsets[(y, x)] = (shift, max(weight, 1e-6))
    return offsets, attempted


def exact_vertical_misalignment(left, right, src_gray, map_left, map_right, shape,
                                src_rgb=None, hdr_scale=None, expected_evidence=None):
    """Localized vertical disparity using exact production source correspondences.

    Returns ``(native_p99_px, normalized_p99_pct, support_pct)``.  The first two values are
    ``None`` when too little unambiguous texture exists, while the independently measured support
    remains available so the policy can distinguish a legitimate N/A from missing evidence.
    Intended horizontal parallax is already represented by each eye's inverse-warp map, so it
    cannot masquerade as vertical disparity. Common vertical motion of both eyes also cancels;
    only the binocular offset is scored.
    """
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape:
        raise ValueError(f"eye geometry differs: {left.shape} != {right.shape}")
    if expected_evidence is None:
        left_evidence = _exact_expected_source_luma(
            left.shape, src_gray, map_left, shape, src_rgb=src_rgb, hdr_scale=hdr_scale)
        right_evidence = _exact_expected_source_luma(
            right.shape, src_gray, map_right, shape, src_rgb=src_rgb, hdr_scale=hdr_scale)
    else:
        left_evidence, right_evidence = expected_evidence
    left_expected, _, left_valid = left_evidence
    right_expected, _, right_valid = right_evidence
    left_offsets, left_attempted = _local_vertical_offsets(
        left, left_expected, left_valid)
    right_offsets, right_attempted = _local_vertical_offsets(
        right, right_expected, right_valid)
    common = sorted(set(left_offsets) & set(right_offsets))
    attempted = min(left_attempted, right_attempted)
    support_pct = 100.0 * len(common) / max(attempted, 1)
    if len(common) < 4 or support_pct < 2.0:
        return None, None, support_pct
    disparities = np.asarray(
        [abs(left_offsets[key][0] - right_offsets[key][0]) for key in common],
        dtype=np.float32)
    weights = np.asarray(
        [np.sqrt(left_offsets[key][1] * right_offsets[key][1]) for key in common],
        dtype=np.float32)
    # A one-percent localized fault influences fewer than five percent of the overlapping tiles,
    # so p95 is structurally blind to exactly the small, salient binocular defect this metric is
    # meant to catch.  P99 remains well below the hard limit on authenticated clean outputs while
    # responding to the controlled one-percent fault ladder.
    p99_analysis_px = weighted_pct(
        disparities, weights, VERTICAL_MISALIGNMENT_QUANTILE)
    # _local_vertical_offsets uses a fixed-height analysis raster above 540p.  Normalize there,
    # then convert back so both the percentage and native-pixel diagnostic are resolution robust.
    analysis_height = max(24, min(left.shape[0], 540))
    p99_pct = p99_analysis_px / max(float(analysis_height), 1.0) * 100.0
    p99_native_px = p99_pct * left.shape[0] / 100.0
    return p99_native_px, p99_pct, support_pct


def weighted_pct(vals, wts, q):
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    c = np.cumsum(wts)
    c /= c[-1]
    return float(np.interp(q, c, vals))


REFERENCE_STREAM_ASPECT = sbs_interocular_metrics.REFERENCE_STREAM_ASPECT


def perceived_disparity_pct(disparity_px, eye_width, eye_height):
    """Disparity as a reference-aspect-equivalent percentage of image geometry.

    Raw percent-of-width changes meaning when the requested image shape changes. Convert through
    eye height and express the result at the validated 5120x2160 reference aspect. No physical
    display size or placement is assumed.
    """
    return sbs_interocular_metrics.perceived_disparity_pct(
        disparity_px, eye_width, eye_height)


def _sample_scalar_uv(image, u, v):
    """D3D-style bilinear/clamped normalized sampling for metric-side depth association."""
    image = np.asarray(image, dtype=np.float32)
    h, w = image.shape
    x = np.clip(np.asarray(u, np.float32) * w - 0.5, 0.0, w - 1.0)
    y = np.clip(np.asarray(v, np.float32) * h - 0.5, 0.0, h - 1.0)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    return ((1.0 - wx) * (1.0 - wy) * image[y0, x0]
            + wx * (1.0 - wy) * image[y0, x1]
            + (1.0 - wx) * wy * image[y1, x0]
            + wx * wy * image[y1, x1]).astype(np.float32)


def _sample_rgb_uv(image, u, v):
    """D3D-style bilinear/clamped normalized sampling for an RGB texture."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"RGB sampler requires HxWx3, got {image.shape}")
    return np.stack([_sample_scalar_uv(image[..., channel], u, v)
                     for channel in range(3)], axis=-1)


def _srgb_to_linear(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92,
                    np.power((rgb + 0.055) / 1.055, 2.4)).astype(np.float32)


def _linear_to_srgb(rgb):
    rgb = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1.0 / 2.4) - 0.055).astype(np.float32)


def _hdr_preview_rgb(linear_rgb):
    """Match the harness's deterministic scRGB-to-PNG diagnostic preview."""
    rgb = np.maximum(np.asarray(linear_rgb, dtype=np.float32), 0.0)
    luminance = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152
                 + rgb[..., 2] * 0.0722)
    rgb = rgb / (1.0 + luminance[..., None])
    peak = np.maximum(1.0, np.max(rgb, axis=-1))
    return _linear_to_srgb(rgb / peak[..., None])


def _largest_component_pct(mask, denominator=None, max_width=256):
    """Largest 4-connected region as a percentage, bounded for offline metric cost.

    Exact per-pixel area is reported separately.  Component size is evaluated on a conservative
    validation raster so a thin but long sword/limb is not erased merely because the deployment
    output is wide.  This is a topology signal, not a replacement for native area.
    """
    mask = np.asarray(mask, dtype=bool)
    denominator = (np.ones(mask.shape, dtype=bool) if denominator is None
                   else np.asarray(denominator, dtype=bool))
    if denominator.shape != mask.shape:
        raise ValueError("component denominator must match its mask")
    if not mask.any() or not denominator.any():
        return 0.0
    scale = min(1.0, max_width / float(mask.shape[1]))
    if scale < 1.0:
        width = max(1, round(mask.shape[1] * scale))
        height = max(1, round(mask.shape[0] * scale))
        mask = np.asarray(Image.fromarray(mask.astype(np.float32), mode="F").resize(
            (width, height), Image.BOX), dtype=np.float32) > 0.0
        denominator = np.asarray(
            Image.fromarray(denominator.astype(np.uint8) * 255, mode="L").resize(
                (width, height), Image.NEAREST), dtype=np.uint8) >= 128
    mask &= denominator
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        stack = [int(start_y * width + start_x)]
        size = 0
        while stack:
            index = stack.pop()
            y, x = divmod(index, width)
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if (0 <= ny < height and 0 <= nx < width and mask[ny, nx]
                        and not visited[ny, nx]):
                    visited[ny, nx] = True
                    stack.append(ny * width + nx)
        largest = max(largest, size)
    return float(largest / max(np.count_nonzero(denominator), 1) * 100.0)


def _inverse_position_jacobian(position, valid, nominal_step):
    """Output-pixel area represented by each sample of an inverted source-U row.

    ``_invert_row`` has already rejected folds, clamps, non-monotonic runs, and ambiguous
    duplicate correspondences.  This second pass refuses to bridge a missing source interval and
    converts the remaining source-uniform samples into output-area weights.  The latter matters
    for comfort tails: a one-source-pixel sample stretched over ten output pixels must contribute
    more rendered area than an ordinary one-pixel sample.
    """
    position = np.asarray(position, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(position)
    weights = np.zeros(position.shape, dtype=np.float32)
    if position.size < 2:
        return weights
    step = np.diff(position)
    adjacent = valid[:-1] & valid[1:] & (step > 1e-6)
    ordinary = step[adjacent]
    typical_step = float(np.median(ordinary)) if ordinary.size else float(nominal_step)
    # Match the monotonic-run inversion's conservative gap rule in output space.  A genuine
    # smooth stretch remains area evidence; an occlusion jump between independent runs does not.
    maximum_step = max(4.0 * float(nominal_step), 8.0 * typical_step)
    usable = adjacent & (step <= maximum_step)
    interval_weight = np.where(usable, step, 0.0).astype(np.float32)
    weights[:-1] += 0.5 * interval_weight
    weights[1:] += 0.5 * interval_weight
    return weights


def _exact_binocular_geometry(mapping, shape, coverage_map=None):
    """Invert both exact eye maps onto common source-U samples.

    The old evaluator doubled each eye's monocular inverse displacement and implicitly assumed a
    perfectly symmetric warp.  That made a common camera translation look like stereo disparity
    and doubled a one-eye fault.  Here the existing monotonic-run inversion is applied to the two
    eyes independently, then actual binocular disparity is measured as ``x_right - x_left`` only
    where the same source point is uniquely and mutually visible.

    Returned ``weight`` is the mean left/right output Jacobian in output-eye pixels.  Raw U outside
    [0, 1], bars, folds, map gaps, ambiguous duplicate runs, and (when supplied) invalid forward
    coverage never receive weight.
    """
    mapping = np.asarray(mapping, dtype=np.float32)
    height = int(shape.get("height", 0))
    eye_width = int(shape.get("eye_width", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if mapping.shape != (height, 2 * eye_width):
        raise ValueError("binocular geometry map does not match its shape contract")
    if not np.isfinite(mapping).all():
        raise ValueError("binocular geometry map contains non-finite coordinates")
    if min(height, eye_width, scale_x, scale_y) <= 0.0:
        raise ValueError("binocular geometry contract is missing positive geometry")
    if coverage_map is not None:
        coverage_map = np.asarray(coverage_map, dtype=bool)
        if coverage_map.shape != mapping.shape:
            raise ValueError("forward coverage does not match binocular geometry")

    lo_x = 0.5 * (1.0 - scale_x)
    lo_y = 0.5 * (1.0 - scale_y)
    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / float(eye_width)
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    content_x = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    content_rows = np.flatnonzero((output_v >= lo_y) & (output_v <= lo_y + scale_y))
    analysis_width = max(2, int(round(scale_x * eye_width)))
    target_u = (np.arange(analysis_width, dtype=np.float32) + 0.5) / analysis_width
    source_v = np.clip((output_v[content_rows] - lo_y) / scale_y, 0.0, 1.0)
    unwarped_x = (lo_x + target_u * scale_x) * eye_width - 0.5
    output_x = np.arange(eye_width, dtype=np.float32)
    nominal_step = scale_x * eye_width / analysis_width

    disparity = np.full((content_rows.size, analysis_width), np.nan, dtype=np.float32)
    symmetry = np.full_like(disparity, np.nan)
    weight = np.zeros_like(disparity)
    support_weight = np.zeros_like(disparity)
    for target_row, output_row in enumerate(content_rows):
        positions = []
        jacobians = []
        for eye_index in range(2):
            offset = eye_index * eye_width
            source_u_row = mapping[output_row, offset:offset + eye_width]
            invertible = content_x.copy()
            if coverage_map is not None:
                invertible &= coverage_map[output_row, offset:offset + eye_width]
            position, unique = sbs_interocular_metrics._invert_row(
                output_x, source_u_row, invertible, target_u)
            positions.append(position)
            jacobians.append(_inverse_position_jacobian(position, unique, nominal_step))
        common = (np.isfinite(positions[0]) & np.isfinite(positions[1])
                  & (jacobians[0] > 0.0) & (jacobians[1] > 0.0))
        if not common.any():
            continue
        disparity[target_row, common] = (
            positions[1][common] - positions[0][common])
        symmetry[target_row, common] = (
            0.5 * (positions[0][common] + positions[1][common])
            - unwarped_x[common])
        weight[target_row, common] = (
            0.5 * (jacobians[0][common] + jacobians[1][common]))
        # Common *rendered area* is limited by the smaller eye footprint. Counting source-grid
        # samples alone lets a map cover the entire source inside a narrow output strip and hide
        # the collapse from the disparity-tail denominator. Keep statistical sample count and
        # output-area support separate.
        support_weight[target_row, common] = np.minimum(
            jacobians[0][common], jacobians[1][common])

    return {
        "target_u": target_u,
        "source_v": source_v,
        "disparity": disparity,
        "symmetry": symmetry,
        "weight": weight,
        "support_weight": support_weight,
        "valid": weight > 0.0,
        "possible_count": int(content_rows.size * analysis_width),
    }


def exact_warp_mapping_metrics(mapping, shape, depth=None, warp_mask=None, tail=0.999,
                               binocular_geometry=None):
    """Geometry metrics from the production shader's exact inverse map.

    Unlike tile phase correlation, this sees smooth/thin/local subjects and does not depend on
    output texture. The old image-derived metrics remain an independent rendered-output sanity
    check; hard comfort and training labels use this exact geometry evidence.
    """
    mapping = np.asarray(mapping, dtype=np.float32)
    if mapping.ndim != 2:
        raise ValueError(f"warp mapping must be scalar Hx(2W), got {mapping.shape}")
    if not np.isfinite(mapping).all():
        raise ValueError("warp mapping contains non-finite coordinates")
    height, packed_width = mapping.shape
    if packed_width % 2:
        raise ValueError(f"warp mapping width must be even, got {mapping.shape}")
    eye_width = packed_width // 2
    if shape.get("width") != packed_width or shape.get("height") != height:
        raise ValueError("warp mapping array does not match its shape contract")
    if shape.get("eye_width") != eye_width or shape.get("eye_height") != height:
        raise ValueError("warp mapping eye geometry is inconsistent")
    source_width = float(shape.get("source_width", 0))
    source_height = float(shape.get("source_height", 0))
    scale_x = float(shape.get("content_scale_x", 0))
    scale_y = float(shape.get("content_scale_y", 0))
    if min(source_width, source_height, scale_x, scale_y) <= 0.0:
        raise ValueError("warp mapping is missing positive source/content geometry")

    if warp_mask is not None:
        warp_mask = np.asarray(warp_mask, dtype=np.float32)
        if warp_mask.shape[:2] != mapping.shape or warp_mask.ndim not in (2, 3):
            raise ValueError(
                f"warp mask {warp_mask.shape} does not match warp mapping {mapping.shape}")
        hole_map = warp_mask if warp_mask.ndim == 2 else warp_mask[..., 0]
        coverage_map = hole_map < 0.5
    else:
        coverage_map = None

    stretch_pcts = []
    fold_pcts = []
    mapping_support_pcts = []
    output_v = (np.arange(height, dtype=np.float32) + 0.5) / float(height)
    lo_y = 0.5 * (1.0 - scale_y)
    content_y = (output_v >= lo_y) & (output_v <= lo_y + scale_y)
    output_u = (np.arange(eye_width, dtype=np.float32) + 0.5) / float(eye_width)
    lo_x = 0.5 * (1.0 - scale_x)
    content_x = (output_u >= lo_x) & (output_u <= lo_x + scale_x)
    content = content_y[:, None] & content_x[None, :]

    for eye_index in range(2):
        sampled_u = mapping[:, eye_index * eye_width:(eye_index + 1) * eye_width]
        valid = content & np.isfinite(sampled_u)
        if not valid.any():
            raise ValueError("warp mapping has no valid content pixels")
        # Topology is measured from the coordinate the live sampler consumes. Off-screen demand
        # is clipped here and therefore appears as a repeated-column stretch instead of a second,
        # redundant clamp percentage.
        live_sampled_u = np.clip(sampled_u, 0.0, 1.0)

        # Exact source-coordinate Jacobian. A healthy unwarped mapping advances by one baseline
        # source step per output pixel. Near-zero/reversed steps expose repeated columns, folds,
        # and rubber-band fill without depending on source texture or a candidate-chosen matcher.
        # Topology measures the coordinate consumed by the texture sampler, not merely requested
        # geometry. Off-screen raw U collapses onto the boundary after main_ps saturates it and
        # must therefore appear as repeated columns/stretch.
        source_x = live_sampled_u * source_width
        source_step = np.diff(source_x, axis=1)
        adjacent = valid[:, 1:] & valid[:, :-1]
        content_adjacent = content[:, 1:] & content[:, :-1]
        mapping_support_pcts.append(float(
            np.count_nonzero(adjacent) / max(np.count_nonzero(content_adjacent), 1) * 100.0))
        baseline_step = source_width / max(scale_x * eye_width, 1.0)
        ratio_map = np.abs(source_step) / max(baseline_step, 1e-6)
        ratio = ratio_map[adjacent]
        if ratio.size:
            stretch_pcts.append(float(np.mean(ratio < 0.35) * 100.0))
            fold_pcts.append(float(np.mean(
                source_step[adjacent] < -0.05 * baseline_step) * 100.0))

    binocular = (binocular_geometry if binocular_geometry is not None else
                 _exact_binocular_geometry(mapping, shape, coverage_map))
    binocular_valid = binocular["valid"]
    binocular_count = int(np.count_nonzero(binocular_valid))
    possible_count = max(int(binocular["possible_count"]), 1)
    binocular_output_area = float(np.sum(binocular["support_weight"]))
    out = {
        "exact_binocular_support_pct": float(
            np.clip(binocular_output_area / possible_count * 100.0, 0.0, 100.0)),
        "exact_binocular_support_count": binocular_count,
        "exact_mapping_support_pct": min(mapping_support_pcts, default=0.0),
        "exact_mapping_stretch_pct": max(stretch_pcts, default=0.0),
        "exact_mapping_fold_pct": max(fold_pcts, default=0.0),
        "exact_local_polarity_support_pct": 0.0,
        "exact_local_polarity_support_count": 0,
    }

    if binocular_count:
        signed = binocular["disparity"][binocular_valid]
        area_weight = binocular["weight"][binocular_valid]
        lo_tail = weighted_pct(signed, area_weight, 1.0 - tail)
        hi_tail = weighted_pct(signed, area_weight, tail)
        one_pixel_pct = perceived_disparity_pct(1.0, eye_width, height)
        disparity_pct = signed * one_pixel_pct
        out.update({
            "exact_positive_disparity_pct": perceived_disparity_pct(
                max(0.0, hi_tail), eye_width, height),
            "exact_negative_disparity_pct": perceived_disparity_pct(
                max(0.0, -lo_tail), eye_width, height),
            "exact_over_3pct_area_pct": float(
                np.sum(area_weight[np.abs(disparity_pct) > 3.0])
                / np.sum(area_weight) * 100.0),
            "exact_symmetry_residual_p95_px": weighted_pct(
                np.abs(binocular["symmetry"][binocular_valid]), area_weight, 0.95),
        })
        out["exact_symmetry_residual_p95_pct"] = perceived_disparity_pct(
            out["exact_symmetry_residual_p95_px"], eye_width, height)

    # The raw binocular sign above is the standard left-to-right displacement xR-xL. Apollo's
    # high-is-near forward warp moves the right eye left and the left eye right, so high-near
    # ordering is positive in -disparity. Keeping that sign conversion explicit prevents the
    # comfort tails (which are directional xR-xL evidence) from being confused with depth order.
    if depth is not None:
        out["exact_polarity_support_pct"] = 0.0
    if depth is not None and binocular_count >= 64:
        target_u = np.broadcast_to(
            binocular["target_u"][None, :], binocular["disparity"].shape)
        source_v_grid = np.broadcast_to(
            binocular["source_v"][:, None], binocular["disparity"].shape)
        sampled_depth = _sample_scalar_uv(depth, target_u, source_v_grid)
        near_signed = -binocular["disparity"]
        area_weight_map = binocular["weight"]
        sampled = sampled_depth[binocular_valid]
        ordered = near_signed[binocular_valid]
        sample_weight = area_weight_map[binocular_valid]
        d20 = weighted_pct(sampled, sample_weight, 0.20)
        d80 = weighted_pct(sampled, sample_weight, 0.80)
        far, near = sampled <= d20, sampled >= d80
        if d80 - d20 >= 0.02 and far.any() and near.any():
            margin = (weighted_pct(ordered[near], sample_weight[near], 0.50)
                      - weighted_pct(ordered[far], sample_weight[far], 0.50))
            out["exact_polarity_ok"] = 100.0 if margin > 0.01 else 0.0
            out["exact_polarity_support_pct"] = float(
                (np.sum(sample_weight[far]) + np.sum(sample_weight[near]))
                / np.sum(sample_weight) * 100.0)
        else:
            out["exact_polarity_ok"] = 100.0

        # Local order must agree too: a globally correct median can hide an inverted face,
        # weapon, or thin subject. Pair tests operate only on mutually visible source samples,
        # never across a rejected fold/gap or an invalid forward-coverage region.
        violation = np.zeros(binocular_valid.shape, dtype=bool)
        local_support = np.zeros(binocular_valid.shape, dtype=bool)
        comparable_count = 0
        grid_height, grid_width = binocular_valid.shape
        for dy, dx in ((0, 1), (0, 4), (1, 0), (4, 0)):
            if dy >= grid_height or dx >= grid_width:
                continue
            a_y = slice(0, grid_height - dy if dy else grid_height)
            b_y = slice(dy, grid_height)
            a_x = slice(0, grid_width - dx if dx else grid_width)
            b_x = slice(dx, grid_width)
            pair_valid = binocular_valid[a_y, a_x] & binocular_valid[b_y, b_x]
            depth_delta = sampled_depth[b_y, b_x] - sampled_depth[a_y, a_x]
            comparable = pair_valid & (np.abs(depth_delta) >= 0.02)
            ordered_delta = ((near_signed[b_y, b_x] - near_signed[a_y, a_x])
                             * np.sign(depth_delta))
            wrong = comparable & (ordered_delta < -0.05)
            comparable_count += int(np.count_nonzero(comparable))
            local_support[a_y, a_x] |= comparable
            local_support[b_y, b_x] |= comparable
            violation[a_y, a_x] |= wrong
            violation[b_y, b_x] |= wrong
        local_support_count = int(np.count_nonzero(local_support))
        out["exact_local_polarity_support_count"] = local_support_count
        out["exact_local_polarity_support_pct"] = float(
            local_support_count / max(binocular_count, 1) * 100.0)
        if comparable_count:
            out["exact_local_polarity_component_pct"] = _largest_component_pct(
                violation, binocular_valid)
    return out


def _horizontal_structure_map(gray, reference_width=3066.0):
    """Horizontal correspondence evidence at several deployment-normalized scales.

    Stereo disparity is horizontal. A horizontal line can have a strong 2-D gradient while
    carrying almost no horizontal matching evidence except at its endpoints, so a generic
    gradient magnitude overstates its visible stereo contribution. This map intentionally uses
    only left/right contrast and preserves both sides of each transition.
    """
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"visible-disparity source must be scalar HxW, got {gray.shape}")
    structure = np.zeros(gray.shape, dtype=np.float32)
    offsets = sorted({
        1,
        max(1, round(4.0 * gray.shape[1] / reference_width)),
        max(1, round(16.0 * gray.shape[1] / reference_width)),
    })
    for offset in offsets:
        if offset >= gray.shape[1]:
            continue
        delta = np.abs(gray[:, offset:] - gray[:, :-offset])
        structure[:, offset:] = np.maximum(structure[:, offset:], delta)
        structure[:, :-offset] = np.maximum(structure[:, :-offset], delta)
    return structure


def exact_visible_disparity_metrics(mapping, shape, src_gray, tail=0.995, warp_mask=None,
                                    binocular_geometry=None):
    """Measure disparity that has independent horizontal image evidence.

    The exact inverse map is authoritative for requested geometry, but geometry in a completely
    textureless region is not by itself proof of visible stereo volume. Weight its signed
    disparity by horizontal structure from the original source. Stereo volume is deliberately a
    style descriptor rather than a reward: a high-tail local-relief score was removed because a
    single disparity spike could improve it while making the picture worse.
    """
    mapping = np.asarray(mapping, dtype=np.float32)
    source = np.asarray(src_gray, dtype=np.float32)
    if not np.isfinite(mapping).all():
        raise ValueError("visible-disparity map contains non-finite coordinates")
    height = int(shape.get("height", 0))
    eye_width = int(shape.get("eye_width", 0))
    source_width = int(shape.get("source_width", 0))
    source_height = int(shape.get("source_height", 0))
    scale_x = float(shape.get("content_scale_x", 0.0))
    scale_y = float(shape.get("content_scale_y", 0.0))
    if mapping.shape != (height, 2 * eye_width):
        raise ValueError("visible-disparity map does not match its shape contract")
    if source.shape != (source_height, source_width):
        raise ValueError(
            f"visible-disparity source {source.shape} does not match "
            f"contract {(source_height, source_width)}")
    if min(height, eye_width, source_width, source_height, scale_x, scale_y) <= 0:
        raise ValueError("visible-disparity contract is missing positive geometry")

    coverage_map = None
    if warp_mask is not None:
        warp_mask = np.asarray(warp_mask, dtype=np.float32)
        if warp_mask.shape[:2] != mapping.shape or warp_mask.ndim not in (2, 3):
            raise ValueError(
                f"warp mask {warp_mask.shape} does not match warp mapping {mapping.shape}")
        hole_map = warp_mask if warp_mask.ndim == 2 else warp_mask[..., 0]
        coverage_map = hole_map < 0.5

    geometry = (binocular_geometry if binocular_geometry is not None else
                _exact_binocular_geometry(mapping, shape, coverage_map))
    valid = geometry["valid"]
    target_u = np.broadcast_to(geometry["target_u"][None, :], valid.shape)
    source_v = np.broadcast_to(geometry["source_v"][:, None], valid.shape)
    sampled_structure = _sample_scalar_uv(
        _horizontal_structure_map(source), target_u, source_v)
    # Two code values of source contrast establish evidence. Above eight code values the sample
    # receives full weight, avoiding domination by compression noise while retaining low-contrast
    # film edges. Geometry's output-Jacobian weight keeps a stretched rendered area proportional.
    evidence_weight = np.clip(sampled_structure * 255.0 / 8.0, 0.0, 1.0)
    supported = valid & (sampled_structure * 255.0 >= 2.0)
    support_count = int(np.count_nonzero(supported))
    content_count = max(int(geometry["possible_count"]), 1)

    out = {
        "exact_visible_support_pct": float(
            support_count / max(content_count, 1) * 100.0),
        "exact_visible_support_count": support_count,
    }
    # Abstain rather than inventing zero when the image has no usable horizontal evidence.
    if support_count < 16:
        return out
    signed = geometry["disparity"][supported]
    signed_weights = geometry["weight"][supported] * evidence_weight[supported]
    lo = weighted_pct(signed, signed_weights, 1.0 - tail)
    hi = weighted_pct(signed, signed_weights, tail)
    out.update({
        "exact_visible_pop_spread_pct": perceived_disparity_pct(
            hi - lo, eye_width, height),
    })
    return out


# --------------------------------------------------------- source-relative metrics

def hdilate(mask, px):
    """Dilate a boolean mask horizontally by +/- px columns."""
    out = mask.copy()
    for s in range(1, px + 1):
        out[:, s:] |= mask[:, :-s]
        out[:, :-s] |= mask[:, s:]
    return out


def dilate2d(mask, px):
    """Dilate a boolean mask in both image axes without wraparound."""
    padded = np.pad(mask, px, mode="constant")
    h, w = mask.shape
    return np.logical_or.reduce([
        padded[px + dy:px + dy + h, px + dx:px + dx + w]
        for dy in range(-px, px + 1) for dx in range(-px, px + 1)
    ])


def resize_to(gray, w, h):
    """Resize a normalized scalar image without silently quantizing it to eight bits."""
    gray = np.asarray(gray, dtype=np.float32)
    if gray.shape == (h, w):
        return gray.copy()
    image = Image.fromarray(gray, mode="F")
    return np.asarray(image.resize((w, h), Image.BILINEAR), dtype=np.float32)


def resize_mask_conservative(mask, w, h):
    """Resize boolean evidence without dropping thin support during downscaling.

    Nearest-neighbour downscaling can sample around a one-pixel silhouette and turn a real edge
    into false-perfect zero support. BOX computes fractional coverage; any covered destination
    pixel remains evidence. Upscaling stays nearest so the mask is not blurred into new support.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == (h, w):
        return mask.copy()
    shrinking = w <= mask.shape[1] and h <= mask.shape[0]
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = np.asarray(image.resize((w, h), Image.BOX if shrinking else Image.NEAREST),
                         dtype=np.float32)
    return resized > 0.0


# Reference eye width the pixel-unit tuning was done at (full-res movie runs). All band/run/reach
# windows scale by (ew / REF_EW) so a metric means the same thing at any output resolution.
REF_EW = 3066.0
# Absolute floor for a "real" silhouette: normalized-depth step per NATIVE depth pixel. Percentile
# thresholds alone always find "edges" (even pure noise on flat content); AND-ing this floor makes
# flat scenes legitimately return zero. Real silhouettes measure ~0.1-0.3/px at depth res.
MIN_DEPTH_STEP = 0.04
HARD_MAX_AGG = {
    "exact_positive_disparity_pct", "exact_negative_disparity_pct",
    "vmisalign_p99_pct",
    "exact_symmetry_residual_p95_pct",
    "exact_mapping_stretch_pct", "exact_mapping_fold_pct",
    "warp_cross_row_shear_severity_pct",
    "experimental_stereo_window_crossed_burden_pct",
    "interocular_phase_orientation_burden_pct",
    "interocular_exposure_rivalry_burden_pct",
    "interocular_color_gain_rivalry_burden_pct",
    "interocular_phase_orientation_evidence_sufficient",
    "interocular_exposure_rivalry_evidence_sufficient",
    "interocular_color_gain_rivalry_evidence_sufficient",
    "exact_local_polarity_component_pct",
    "source_coverage_worst_patch_bad_pct", "image_integrity_worst_patch_bad_pct",
}
HARD_MIN_AGG = {
    "exact_binocular_support_pct", "source_coverage_pct", "image_integrity_pct",
    "exact_polarity_ok", "depth_gt_polarity_ok",
}


def eye_scale(ew):
    return ew / REF_EW


def silhouette_edges(depth, ew, eh, edge_pct=99.3):
    """Depth-silhouette mask at eye resolution, resolution-independently: the gradient test runs at
    the NATIVE depth resolution (constant ~602x336 regardless of clip size), with an absolute
    depth-step floor AND'd with the percentile, then the boolean mask is nearest-upsampled."""
    gx_d = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    thr = max(float(np.percentile(gx_d, edge_pct)), MIN_DEPTH_STEP)
    edge_d = gx_d >= thr
    if not edge_d.any():
        return np.zeros((eh, ew), bool)
    return resize_mask_conservative(edge_d, ew, eh)


def _hopen(a, r):
    """Horizontal grayscale opening (erode then dilate) with radius r -- removes bright features
    narrower than 2r+1 px, leaving the broad fg/bg. eye - open = a horizontal white top-hat."""
    a = np.asarray(a)
    if r <= 0:
        return a.copy()

    def window(x, reducer):
        h, w = x.shape
        padded = np.pad(x, ((0, 0), (r, r)), mode="edge")
        return reducer([padded[:, o:o + w] for o in range(2 * r + 1)])

    return window(window(a, np.minimum.reduce), np.maximum.reduce)


def _shift_x_edge(a, shift):
    """Shift an image horizontally without wraparound; repeat the nearest border value."""
    if shift == 0:
        return a
    out = np.empty_like(a)
    if shift > 0:
        out[:, shift:] = a[:, :-shift]
        out[:, :shift] = a[:, :1]
    else:
        n = -shift
        out[:, :-n] = a[:, n:]
        out[:, -n:] = a[:, -1:]
    return out


def _box3(a):
    """3x3 edge-padded mean used to make source matching respond to patches, not lone pixels."""
    p = np.pad(a, ((1, 1), (1, 1)), mode="edge")
    integral = np.pad(p, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    total = (integral[3:, 3:] - integral[:-3, 3:]
             - integral[3:, :-3] + integral[:-3, :-3])
    return total / 9.0


def source_align_map(eye, src_gray, max_shift=None, return_shift=False):
    """Regularized source patch error and selected source sample for every output pixel.

    A completely independent winner at every pixel can assemble an impossible output from
    unrelated source locations and hide stretch/halo artifacts. The second pass mildly penalizes
    shifts that disagree with the local five-pixel median while preserving real disparity steps.
    """
    eh, ew = eye.shape
    src = resize_to(src_gray, ew, eh)
    radius = max_shift if max_shift is not None else max(4, round(40 * eye_scale(ew)))
    shifts = np.arange(-radius, radius + 1, dtype=np.int16)
    costs = []
    candidates = []
    for shift in shifts:
        candidate = _shift_x_edge(src, shift)
        candidates.append(candidate)
        costs.append(_box3(np.abs(eye - candidate)).astype(np.float32))
    costs = np.stack(costs)
    candidates = np.stack(candidates)
    first = np.argmin(costs, axis=0)
    first_shift = shifts[first]
    neighborhood = np.stack([_shift_x_edge(first_shift, dx) for dx in (-2, -1, 0, 1, 2)])
    local_shift = np.median(neighborhood, axis=0)
    regularized = costs + (2.0 / 255.0) * np.abs(
        shifts[:, None, None].astype(np.float32) - local_shift[None, :, :])
    selected = np.argmin(regularized, axis=0)
    best = np.take_along_axis(costs, selected[None, :, :], axis=0)[0]
    aligned = np.take_along_axis(candidates, selected[None, :, :], axis=0)[0]
    if return_shift:
        return best, aligned, shifts[selected].astype(np.float32), radius
    return best, aligned, radius


def source_relative_metrics(eye, src_gray, max_shift=None,
                            coverage_error=4.0 / 255.0):
    """Fallback source-relative renderer integrity for a dump without an exact map.

    Horizontal source search makes intended stereo displacement free. Coverage measures how much
    of the interior can still be explained by source content, while integrity measures retention
    of real source texture. This permissive matcher is diagnostic only; harness evaluation uses
    ``exact_source_relative_metrics`` and the exact production map.
    """
    original_h, original_w = eye.shape
    scale = min(1.0, 256.0 / original_w)
    if scale < 1.0:
        ew, eh = round(original_w * scale), round(original_h * scale)
        eye = resize_to(eye, ew, eh)
        if max_shift is not None:
            max_shift = max(1, round(max_shift * scale))
    else:
        eh, ew = eye.shape
    best, aligned, radius = source_align_map(eye, src_gray, max_shift)
    valid = np.ones_like(eye, dtype=bool)
    if ew > 2 * radius:
        valid[:, :radius] = False
        valid[:, ew - radius:] = False
    interior = best[valid]
    out = {
        "source_residual_p50": float(np.percentile(interior, 50) * 255.0),
        "source_residual_p95": float(np.percentile(interior, 95) * 255.0),
        "source_coverage_pct": float(np.mean(interior <= coverage_error) * 100.0),
    }

    gx_eye = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    gy_eye = np.abs(np.diff(eye, axis=0, prepend=eye[:1, :]))
    gx_src = np.abs(np.diff(aligned, axis=1, prepend=aligned[:, :1]))
    gy_src = np.abs(np.diff(aligned, axis=0, prepend=aligned[:1, :]))
    texture_src = np.hypot(gx_src, gy_src)
    texture_eye = np.hypot(gx_eye, gy_eye)
    textured = valid & (texture_src >= 4.0 / 255.0)
    if textured.any():
        ratio = texture_eye[textured] / np.maximum(texture_src[textured], 1e-6)
        cosine = ((gx_eye[textured] * gx_src[textured] + gy_eye[textured] * gy_src[textured]) /
                  np.maximum(texture_eye[textured] * texture_src[textured], 1e-6))
        out["image_integrity_pct"] = float(
            np.mean((ratio >= 0.5) & (ratio <= 2.0) & (cosine >= 0.5)) * 100.0)
    else:
        out["image_integrity_pct"] = 100.0

    return out


def hdist_x(source, max_distance):
    """Horizontal distance to the nearest true sample, capped at ``max_distance``."""
    source = np.asarray(source, dtype=bool)
    distance = np.full(source.shape, max_distance, np.float32)
    distance[source] = 0.0
    grown = source.copy()
    for radius in range(1, max_distance + 1):
        grown = hdilate(grown, 1)
        distance[(distance == max_distance) & grown] = radius
    return distance


def _local_mean(values, radius):
    """Edge-padded local mean without an optional image-processing dependency."""
    values = np.asarray(values, dtype=np.float32)
    if radius <= 0:
        return values.copy()
    size = 2 * radius + 1
    horizontal = np.zeros(values.shape, dtype=np.float32)
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    for offset in range(size):
        horizontal += padded[:, offset:offset + values.shape[1]]
    vertical = np.zeros(values.shape, dtype=np.float32)
    padded = np.pad(horizontal, ((radius, radius), (0, 0)), mode="edge")
    for offset in range(size):
        vertical += padded[offset:offset + values.shape[0], :]
    return vertical / float(size * size)


def exact_image_integrity_maps(eye, expected, valid, radius=2):
    """Return localized source-texture support and corruption maps.

    A point-gradient ratio is unstable at anti-aliased edges and sees blur only at the few pixels
    where a first difference happens to cross the source threshold.  Instead, compare 5x5 local
    RMS gradient energy and the RMS gradient-vector residual.  The energy ratio catches lost
    texture and ringing, while the vector residual catches equal-energy phase/orientation errors.
    Fixed code-value floors keep the meaning independent of unrelated image contrast.

    The helper is the single detector-map contract used by both the scalar evaluator and the
    actual-output corruption validator.  ``bad`` is meaningful only where ``textured`` is true.
    """
    eye = np.asarray(eye, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if eye.shape != expected.shape or eye.shape != valid.shape:
        raise ValueError(
            f"integrity evidence geometry differs: eye={eye.shape}, "
            f"expected={expected.shape}, valid={valid.shape}")
    if not (np.isfinite(eye).all() and np.isfinite(expected).all()):
        raise ValueError("integrity evidence contains non-finite values")
    if radius < 1:
        raise ValueError("integrity detector radius must be positive")

    gx_eye = np.diff(eye, axis=1, prepend=eye[:, :1])
    gy_eye = np.diff(eye, axis=0, prepend=eye[:1, :])
    gx_expected = np.diff(expected, axis=1, prepend=expected[:, :1])
    gy_expected = np.diff(expected, axis=0, prepend=expected[:1, :])

    expected_energy = np.sqrt(np.maximum(
        _local_mean(gx_expected * gx_expected + gy_expected * gy_expected, radius), 0.0))
    eye_energy = np.sqrt(np.maximum(
        _local_mean(gx_eye * gx_eye + gy_eye * gy_eye, radius), 0.0))
    vector_error = np.sqrt(np.maximum(_local_mean(
        (gx_eye - gx_expected) ** 2 + (gy_eye - gy_expected) ** 2, radius), 0.0))

    # A local window crossing a bar/map hole has no exact reference contract. Exclude it instead
    # of allowing invalid neighbours to create artificial edge energy at the support boundary.
    interior = _local_mean(valid.astype(np.float32), radius) >= 1.0 - 1e-6

    texture_floor = 3.0 / 255.0
    textured = interior & (expected_energy >= texture_floor)
    energy_ratio = eye_energy / np.maximum(expected_energy, 1e-6)
    relative_vector_error = vector_error / np.maximum(expected_energy, texture_floor)
    # Exact source reconstruction is floating point, while harness PNG evidence is quantized.
    # Near the texture floor, a sub-code-value rounding residual can be a large *relative*
    # gradient error and make a handful of perfectly rendered edge pixels look corrupt. Require
    # more than the two-axis RMS of one 8-bit code step before a relative failure is actionable.
    # Real blur/ringing at the 3-code texture floor still clears this absolute guard.
    absolute_gradient_error_floor = 1.5 / 255.0
    energy_bad = (
        ((energy_ratio < 0.78) | (energy_ratio > 1.28))
        & (np.abs(eye_energy - expected_energy) > absolute_gradient_error_floor))
    vector_bad = (
        (relative_vector_error > 0.25)
        & (vector_error > absolute_gradient_error_floor))
    bad = textured & (energy_bad | vector_bad)
    return bad, textured


def worst_local_bad_fraction(bad, support, radius=None, min_support_fraction=0.20):
    """Worst supported local defect fraction on a resolution-scaled visual patch."""
    bad = np.asarray(bad, dtype=bool)
    support = np.asarray(support, dtype=bool)
    if bad.shape != support.shape or bad.ndim != 2:
        raise ValueError("localized defect/support maps must be matching HxW arrays")
    if radius is None:
        radius = max(3, int(round(min(bad.shape) / 100.0)))
    support_fraction = _local_mean(support.astype(np.float32), radius)
    bad_fraction = _local_mean((bad & support).astype(np.float32), radius)
    eligible = support_fraction >= float(min_support_fraction)
    if not np.any(eligible):
        return None
    local_fraction = np.divide(
        bad_fraction, support_fraction, out=np.zeros_like(bad_fraction),
        where=support_fraction > 1e-6)
    return float(np.max(local_fraction[eligible]) * 100.0)


def exact_source_relative_metrics(eye, src_gray, sampled_source_u, shape,
                                  coverage_error=4.0 / 255.0, eye_rgb=None, src_rgb=None,
                                  hdr_scale=None, expected_evidence=None):
    """Source fidelity using the exact source coordinate consumed by the production shader.

    The former matcher selected whichever nearby source patch best explained each candidate
    output. That was useful as a permissive smoke test, but unsafe as a label: a distorted warp
    could change the selected correspondence or erase its own support. This path reconstructs the
    source sample from the harness sidecar, evaluates at native output resolution, and exposes
    attempted support explicitly. Baked source rims/specular edges cancel because they are present
    in both the source sample and the output; only new bright/dark residuals remain.
    """
    eye = np.asarray(eye, dtype=np.float32)
    if (eye_rgb is None) != (src_rgb is None):
        raise ValueError("exact source RGB evidence requires both output and source RGB")
    if expected_evidence is None:
        expected, expected_rgb, valid = _exact_expected_source_luma(
            eye.shape, src_gray, sampled_source_u, shape,
            src_rgb=src_rgb, hdr_scale=hdr_scale)
    else:
        expected, expected_rgb, valid = expected_evidence
    if eye_rgb is not None:
        eye_rgb = np.asarray(eye_rgb, dtype=np.float32)
        if eye_rgb.shape != eye.shape + (3,):
            raise ValueError(f"output RGB {eye_rgb.shape} does not match luma {eye.shape}")
        eye = rgb_luma(eye_rgb)
    residual = np.abs(eye - expected)
    values = residual[valid] * 255.0
    coverage_bad = valid & (residual > coverage_error)
    out = {
        "source_residual_p50": float(np.percentile(values, 50)),
        "source_residual_p95": float(np.percentile(values, 95)),
        "source_coverage_pct": float(np.mean(values <= coverage_error * 255.0) * 100.0),
        "source_fidelity_support_pct": float(np.mean(valid) * 100.0),
    }
    local_coverage = worst_local_bad_fraction(
        coverage_bad, valid, min_support_fraction=0.80)
    if local_coverage is not None:
        out["source_coverage_worst_patch_bad_pct"] = local_coverage
    if expected_rgb is not None:
        rgb_delta = np.abs(eye_rgb - expected_rgb)
        color_error = np.max(rgb_delta, axis=-1)[valid] * 255.0
        out["source_color_residual_p95"] = float(np.percentile(color_error, 95))

    integrity_bad, textured = exact_image_integrity_maps(eye, expected, valid)
    out["image_integrity_support"] = float(np.mean(textured) * 100.0)
    if textured.any():
        out["image_integrity_pct"] = float(
            np.mean(~integrity_bad[textured]) * 100.0)
        local_integrity = worst_local_bad_fraction(
            integrity_bad, textured, min_support_fraction=0.20)
        if local_integrity is not None:
            out["image_integrity_worst_patch_bad_pct"] = local_integrity

    return out


def resize_depth(depth, w, h):
    """Float depth resize without the 8-bit quantization used by resize_to()."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape == (h, w):
        return depth.copy()
    im = Image.fromarray(depth, mode="F")
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32)


def resize_metric_depth(depth, w, h, validity=None):
    """Resize metric depth without bleeding invalid zero/NaN pixels into valid inverse depth.

    Bilinear interpolation of a valid depth beside zero creates a tiny positive value. Inverting
    that value produces an arbitrarily large false GT error, especially on real RGB-D silhouette
    holes. Interpolate values and validity weights separately, normalize, and accept only pixels
    whose bilinear footprint was effectively all valid.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    if validity is not None:
        validity = np.asarray(validity, dtype=bool)
        if validity.shape != depth.shape:
            raise ValueError(
                f"GT validity {validity.shape} does not match depth {depth.shape}")
        valid &= validity
    values = resize_depth(np.where(valid, depth, 0.0), w, h)
    weights = resize_depth(valid.astype(np.float32), w, h)
    resized = np.divide(values, weights, out=np.zeros_like(values), where=weights > 1e-6)
    return resized, weights >= 0.999


def resize_disparity_ground_truth(disparity, w, h, validity=None):
    """Resize disparity while keeping invalid/occluded samples out of interpolation."""
    disparity = np.asarray(disparity, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity >= 0.0)
    if validity is not None:
        validity = np.asarray(validity, dtype=bool)
        if validity.shape != disparity.shape:
            raise ValueError(
                f"GT validity {validity.shape} does not match disparity {disparity.shape}")
        valid &= validity
    values = resize_depth(np.where(valid, disparity, 0.0), w, h)
    weights = resize_depth(valid.astype(np.float32), w, h)
    resized = np.divide(values, weights, out=np.zeros_like(values), where=weights > 1e-6)
    return resized, weights >= 0.999


def align_relative_depth(prediction, target, valid):
    """Affine-align relative disparity without allowing a polarity inversion.

    A negative scale is not a monocular scale ambiguity: it swaps near and far. Collapse such a
    fit to the best constant prediction so both structure and edge metrics reject the inversion.
    Returns ``(aligned, scale)``; flat targets use shift-only alignment and scale 1.
    """
    pred = np.asarray(prediction, np.float32)
    pv, tv = pred[valid], target[valid]
    t5, t95 = np.percentile(tv, (5, 95))
    if t95 - t5 < 1e-4:
        return pred + float(np.median(tv) - np.median(pv)), 1.0

    def weighted_fit(weights):
        weights = np.asarray(weights, np.float64)
        pred64, target64 = pv.astype(np.float64), tv.astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-9:
            return None
        pred_mean = float(np.sum(weights * pred64) / weight_sum)
        target_mean = float(np.sum(weights * target64) / weight_sum)
        variance = float(np.sum(weights * (pred64 - pred_mean) ** 2))
        if variance <= 1e-12:
            return 0.0, target_mean
        fitted_scale = float(np.sum(
            weights * (pred64 - pred_mean) * (target64 - target_mean)) / variance)
        return fitted_scale, target_mean - fitted_scale * pred_mean

    scale, shift = weighted_fit(np.ones_like(pv, dtype=np.float64))
    if scale <= 0.0:
        return np.full_like(pred, float(np.median(tv))), float(scale)
    # Robust positive IRLS: isolated invalid/reflective GT values must not rotate the global
    # affine fit and turn a good relative-depth map into a bad training/evaluation target.
    for _ in range(4):
        residual = tv.astype(np.float64) - (scale * pv.astype(np.float64) + shift)
        center = float(np.median(residual))
        sigma = 1.4826 * float(np.median(np.abs(residual - center))) + 1e-9
        cutoff = 1.5 * sigma
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual - center), 1e-12))
        fitted = weighted_fit(weights)
        if fitted is None or fitted[0] <= 0.0:
            break
        scale, shift = fitted
    return pred * float(scale) + float(shift), float(scale)


def prepare_depth_ground_truth(prediction, ground_truth, kind="disparity", validity=None):
    """Return polarity-preserving alignment, validity, range, normalization, and fitted scale."""
    pred = np.asarray(prediction, np.float32)
    if kind in ("metric", "depth"):
        gt, valid = resize_metric_depth(
            ground_truth, pred.shape[1], pred.shape[0], validity)
    else:
        gt, valid = resize_disparity_ground_truth(
            ground_truth, pred.shape[1], pred.shape[0], validity)
    if kind in ("metric", "depth"):
        target = np.zeros_like(gt)
        target[valid] = 1.0 / gt[valid]
    else:
        valid &= gt >= 0.0
        target = gt
    if valid.sum() < max(64, int(np.ceil(pred.size * 0.05))):
        return None

    tv = target[valid]
    t5, t95 = np.percentile(tv, (5, 95))
    trange = float(t95 - t5)
    if trange < 1e-4:
        aligned, scale = align_relative_depth(pred, target, valid)
        norm = 1.0
    else:
        aligned, scale = align_relative_depth(pred, target, valid)
        norm = trange
    return aligned, target, valid, trange, norm, scale


def depth_ground_truth_edges(aligned, target, valid, trange, threshold_factor=0.08):
    """Return range-normalized prediction and GT boundaries.

    Metric-depth inputs are converted to inverse metres, whose absolute scale varies drastically
    with scene distance. Normalize both fields by the authenticated robust GT range before taking
    gradients so the edge contract is dimensionless instead of silently erasing distant edges.
    """
    edge_scale = max(float(trange), 1e-6)
    normalized_target = target / edge_scale
    normalized_aligned = aligned / edge_scale
    gx_t = np.abs(np.diff(normalized_target, axis=1, prepend=normalized_target[:, :1]))
    gy_t = np.abs(np.diff(normalized_target, axis=0, prepend=normalized_target[:1, :]))
    gx_p = np.abs(np.diff(normalized_aligned, axis=1, prepend=normalized_aligned[:, :1]))
    gy_p = np.abs(np.diff(normalized_aligned, axis=0, prepend=normalized_aligned[:1, :]))
    # A gradient is valid only when both samples used to form it are valid. Otherwise the edge of
    # a missing/zero metric-depth region is incorrectly scored as scene geometry.
    valid_x = valid & np.concatenate((valid[:, :1], valid[:, :-1]), axis=1)
    valid_y = valid & np.concatenate((valid[:1, :], valid[:-1, :]), axis=0)
    edge_threshold = float(threshold_factor)
    gt_edge = ((valid_x & (gx_t >= edge_threshold)) |
               (valid_y & (gy_t >= edge_threshold)))
    pred_edge = ((valid_x & (gx_p >= edge_threshold)) |
                 (valid_y & (gy_p >= edge_threshold)))
    return pred_edge, gt_edge


def _boundary_f1(pred_edge, gt_edge, tolerance):
    if not gt_edge.any():
        return 100.0 if not pred_edge.any() else 0.0
    if not pred_edge.any():
        return 0.0
    gt_near = dilate2d(gt_edge, tolerance)
    pred_near = dilate2d(pred_edge, tolerance)
    precision = float(np.mean(gt_near[pred_edge]))
    recall = float(np.mean(pred_near[gt_edge]))
    return 200.0 * precision * recall / max(precision + recall, 1e-9)


def depth_ground_truth_metrics(prediction, ground_truth, kind="disparity", validity=None):
    """Positive-affine-aligned relative-depth accuracy plus boundary accuracy.

    Monocular models predict relative disparity, so comparing raw values to metric depth is not
    meaningful. Metric depth is converted to inverse depth, then prediction is affine-aligned on
    valid pixels. Constant-GT scenes use shift-only alignment so hallucinated structure cannot be
    fitted away. The RMSE is normalized by GT robust range (or full normalized range for flat GT).
    """
    prepared = prepare_depth_ground_truth(prediction, ground_truth, kind, validity)
    if prepared is None:
        return None
    aligned, target, valid, trange, norm, scale = prepared
    tv = target[valid]
    error = aligned[valid] - tv
    affine_nrmse_pct = float(
        np.sqrt(np.mean(error * error)) / max(norm, 1e-6) * 100.0)
    # Keep the strict one-pixel boundary contract as the sole compact edge score. Averaging it
    # with two/coarse three-pixel tolerances lets a broad, misplaced boundary buy back a real 1 px
    # regression—the opposite of the thin-silhouette behavior this evaluator must protect.
    pred_edge, gt_edge = depth_ground_truth_edges(
        aligned, target, valid, trange, threshold_factor=0.04)
    strict_edge_f1 = _boundary_f1(pred_edge, gt_edge, tolerance=1)
    return {
        "depth_gt_affine_nrmse_pct": affine_nrmse_pct,
        "depth_gt_valid_pct": float(np.mean(valid) * 100.0),
        # A negative affine fit is a polarity inversion, not a monocular scale ambiguity.  The
        # alignment already collapses it to a constant so accuracy metrics fail too; retain this
        # explicit, non-cancellable signal for audits and hard qualification.
        "depth_gt_polarity_ok": 100.0 if scale > 0.0 else 0.0,
        "depth_gt_edge_f1": strict_edge_f1,
    }


def depth_ground_truth_lag(prediction, ground_truth, previous_ground_truth,
                           kind="disparity", validity=None,
                           previous_validity=None):
    """Positive boundary-F1 advantage for the previous GT frame over the current GT frame.

    Current depth should match current-frame geometry at least as well as the previous frame.
    Held depth on moving content instead matches the previous silhouette better. Clamp at zero
    so unrelated prediction noise cannot cancel real lag in other frames.
    """
    current = depth_ground_truth_metrics(prediction, ground_truth, kind, validity)
    previous = depth_ground_truth_metrics(
        prediction, previous_ground_truth, kind, previous_validity)
    if current is None or previous is None:
        return None
    return max(0.0, previous["depth_gt_edge_f1"] - current["depth_gt_edge_f1"])


def static_region_mask(src, prev_src, ew, eh, motion_threshold=3.0 / 255.0):
    """Source-static eye-resolution mask with disparity-radius exclusion around motion."""
    now = resize_to(src, ew, eh)
    before = resize_to(prev_src, ew, eh)
    moving = np.abs(now - before) >= motion_threshold
    radius = max(4, round(40 * eye_scale(ew)))
    return ~hdilate(moving, radius)


def static_region_jitter(left, right, prev_left, prev_right, src, prev_src,
                         motion_threshold=3.0 / 255.0, min_support=TEMPORAL_MIN_SUPPORT):
    """Worst-eye temporal change over source regions that did not move.

    Source-motion pixels are horizontally dilated by the normal disparity radius before exclusion.
    Inside retained regions, subtract the *signed* matched mono-source code-value change before
    taking the residual magnitude. This removes film grain, codec noise, and tiny exposure drift
    that the output reproduces, without allowing an equal-and-opposite output change to cancel.
    Returns (p95 luma/255, stable support fraction), or (None, support) when too little static
    evidence remains (camera motion / scene cut).
    """
    eh, ew = left.shape
    stable = static_region_mask(src, prev_src, ew, eh, motion_threshold)
    support = float(stable.mean())
    if support < min_support:
        return None, support
    source_delta = resize_to(src, ew, eh) - resize_to(prev_src, ew, eh)
    left_delta = np.abs((left - prev_left) - source_delta)[stable] * 255.0
    right_delta = np.abs((right - prev_right) - source_delta)[stable] * 255.0
    return max(float(np.percentile(left_delta, 95)),
               float(np.percentile(right_delta, 95))), support


def _tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    return sorted(set(range(0, length - tile + 1, stride)) | {length - tile})


def dense_source_flow(prev_src, src, ew, eh, tile=64, stride=32, min_var=2e-4):
    """Classical dense optical flow from overlapping phase-correlated source tiles.

    The returned vector maps previous -> current coordinates. Overlap/Hann accumulation produces
    a dense field while the downstream photometric mask rejects boundary tiles and ambiguous flow.
    This intentionally has no model/runtime dependency beyond NumPy.
    """
    before = resize_to(prev_src, ew, eh)
    now = resize_to(src, ew, eh)
    tile = max(16, min(tile, ew, eh))
    stride = max(8, min(stride, tile // 2))
    acc_u = np.zeros((eh, ew), np.float32)
    acc_v = np.zeros((eh, ew), np.float32)
    acc_w = np.zeros((eh, ew), np.float32)
    hann = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 0.05
    for y in _tile_positions(eh, tile, stride):
        for x in _tile_positions(ew, tile, stride):
            a = now[y:y + tile, x:x + tile]
            b = before[y:y + tile, x:x + tile]
            variance = max(float(a.var()), float(b.var()))
            if variance < min_var:
                continue
            dy, dx = phase_shift(a, b)  # shift previous tile onto current tile
            if abs(dx) > tile * 0.4 or abs(dy) > tile * 0.4:
                continue
            confidence = min(1.0, variance / 0.01)
            weight = hann * confidence
            acc_u[y:y + tile, x:x + tile] += dx * weight
            acc_v[y:y + tile, x:x + tile] += dy * weight
            acc_w[y:y + tile, x:x + tile] += weight
    valid = acc_w > 1e-5
    u = np.divide(acc_u, acc_w, out=np.zeros_like(acc_u), where=valid)
    v = np.divide(acc_v, acc_w, out=np.zeros_like(acc_v), where=valid)
    return u, v, valid


def warp_previous_with_flow(previous, u, v):
    """Bilinearly sample a previous frame into current coordinates using prev->current flow."""
    h, w = previous.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = xx - u, yy - v
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    x0 = np.floor(np.clip(sx, 0, w - 1)).astype(np.int32)
    y0 = np.floor(np.clip(sy, 0, h - 1)).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx, wy = sx - x0, sy - y0
    sampled = ((1 - wx) * (1 - wy) * previous[y0, x0]
               + wx * (1 - wy) * previous[y0, x1]
               + (1 - wx) * wy * previous[y1, x0]
               + wx * wy * previous[y1, x1])
    return sampled.astype(np.float32), valid


def warp_previous_nearest_with_flow(previous, u, v):
    """Edge-preserving nearest sample of a previous scalar field into current coordinates.

    Color/photometric validation benefits from bilinear filtering, but applying it to depth
    invents intermediate values at silhouettes and widens warp halos. A live depth-transport
    shader can implement this exact point-sampled operation cheaply.
    """
    h, w = previous.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = xx - u, yy - v
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    xi = np.rint(np.clip(sx, 0, w - 1)).astype(np.int32)
    yi = np.rint(np.clip(sy, 0, h - 1)).astype(np.int32)
    return previous[yi, xi].astype(np.float32), valid


def resize_forward_flow_to_current(flow, valid, width, height):
    """Resize source-grid prev->current flow and splat it onto the current-frame grid.

    Public optical-flow ground truth is conventionally indexed at previous-frame pixels, while
    ``warp_previous_with_flow`` needs a vector at each current pixel. Nearest forward splatting
    performs that coordinate conversion without pretending collisions/holes are valid evidence.
    """
    flow = np.asarray(flow, dtype=np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"optical flow must be HxWx2, got {flow.shape}")
    sh, sw = flow.shape[:2]
    valid = np.asarray(valid, dtype=bool) if valid is not None else np.isfinite(flow).all(axis=2)
    if valid.shape != (sh, sw):
        raise ValueError(f"optical-flow valid mask {valid.shape} does not match {(sh, sw)}")

    def scale_plane(plane, factor):
        image = Image.fromarray(np.nan_to_num(plane).astype(np.float32), mode="F")
        return np.asarray(image.resize((width, height), Image.BILINEAR), np.float32) * factor
    fu = scale_plane(flow[..., 0], width / float(sw))
    fv = scale_plane(flow[..., 1], height / float(sh))
    vm = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize(
        (width, height), Image.NEAREST)) > 127
    yy, xx = np.mgrid[:height, :width]
    tx = np.rint(xx + fu).astype(np.int32)
    ty = np.rint(yy + fv).astype(np.int32)
    keep = vm & np.isfinite(fu) & np.isfinite(fv)
    keep &= (tx >= 0) & (tx < width) & (ty >= 0) & (ty < height)
    flat = (ty[keep] * width + tx[keep]).ravel()
    sum_u = np.zeros(height * width, np.float32)
    sum_v = np.zeros(height * width, np.float32)
    weight = np.zeros(height * width, np.float32)
    np.add.at(sum_u, flat, fu[keep])
    np.add.at(sum_v, flat, fv[keep])
    np.add.at(weight, flat, 1.0)
    occupied = weight > 0
    u = np.divide(sum_u, weight, out=np.zeros_like(sum_u), where=occupied).reshape(height, width)
    v = np.divide(sum_v, weight, out=np.zeros_like(sum_v), where=occupied).reshape(height, width)
    return u, v, occupied.reshape(height, width)


def flow_temporal_metrics(left, right, prev_left, prev_right, src, prev_src,
                          depth=None, prev_depth=None, min_support=TEMPORAL_MIN_SUPPORT,
                          reference_flow=None, reference_valid=None):
    """Motion-compensated output/depth temporal error.

    Exact public-dataset flow is used when supplied; otherwise the deterministic classical
    source-image estimator remains the fallback for ordinary clips.
    """
    eh, ew = left.shape
    # Quarter-area validation is sufficient for temporal structure and keeps the deterministic
    # evaluator fast. Values remain luma/255; flow vectors stay in validation-pixel units.
    scale = min(1.0, 256.0 / ew)
    vw, vh = max(32, round(ew * scale)), max(24, round(eh * scale))
    cur_l, cur_r = resize_to(left, vw, vh), resize_to(right, vw, vh)
    old_l, old_r = resize_to(prev_left, vw, vh), resize_to(prev_right, vw, vh)
    if reference_flow is not None:
        u, v, flow_valid = resize_forward_flow_to_current(
            reference_flow, reference_valid, vw, vh)
    else:
        u, v, flow_valid = dense_source_flow(prev_src, src, vw, vh)
    now_src = resize_to(src, vw, vh)
    before_src = resize_to(prev_src, vw, vh)
    warped_src, warp_valid = warp_previous_with_flow(before_src, u, v)
    reliable = flow_valid & warp_valid & (np.abs(now_src - warped_src) <= 10.0 / 255.0)
    # Shrink reliable regions so tile-flow boundary errors do not masquerade as output flicker.
    reliable &= ~hdilate(~reliable, 1)
    support = float(reliable.mean())
    if support < min_support:
        return None, None, support

    prev_l_warp, l_valid = warp_previous_with_flow(old_l, u, v)
    prev_r_warp, r_valid = warp_previous_with_flow(old_r, u, v)
    mask = reliable & l_valid & r_valid
    # Flow interpolation/illumination change is already visible in the registered mono source and
    # must not be labeled as a stereo-pipeline defect. Subtract that change while it is still
    # signed, then take the residual magnitude. Subtracting magnitudes would incorrectly score an
    # equal-and-opposite output change as perfect temporal agreement.
    source_delta = now_src - warped_src
    lerr = np.abs((cur_l - prev_l_warp) - source_delta)[mask] * 255.0
    rerr = np.abs((cur_r - prev_r_warp) - source_delta)[mask] * 255.0
    output_p95 = max(float(np.percentile(lerr, 95)), float(np.percentile(rerr, 95)))

    depth_p95 = None
    if depth is not None and prev_depth is not None:
        cur_d = resize_depth(depth, vw, vh)
        old_d = resize_depth(prev_depth, vw, vh)
        prev_d_warp, d_valid = warp_previous_nearest_with_flow(old_d, u, v)
        dmask = reliable & d_valid
        if float(dmask.mean()) >= min_support:
            depth_p95 = float(np.percentile(np.abs(cur_d - prev_d_warp)[dmask], 95) * 255.0)
    return output_p95, depth_p95, support


# ----------------------------------------------------------------------- per-frame

def measure(dump_dir):
    sbs_p = os.path.join(dump_dir, "sbs.png")
    source_p = os.path.join(dump_dir, "source.png")
    if not os.path.exists(sbs_p):
        return None
    sbs = load_gray(sbs_p)
    left, right = split_eyes(sbs)

    out = {}
    field = disparity_field(left, right)
    if field is not None:
        _, dys, weights = field
        out["vmisalign_p99_px"] = weighted_pct(
            np.abs(dys), weights, VERTICAL_MISALIGNMENT_QUANTILE)
        out["vmisalign_p99_pct"] = out["vmisalign_p99_px"] / left.shape[0] * 100.0

    if os.path.exists(source_p):
        src = load_gray(source_p)
        lm, rm = source_relative_metrics(left, src), source_relative_metrics(right, src)
        for key in ("source_coverage_pct", "image_integrity_pct"):
            out[key] = min(lm[key], rm[key])

    model = ""
    meta_p = os.path.join(dump_dir, "meta.txt")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            for line in f:
                if line.startswith("depth_model="):
                    model = line.strip().split("=", 1)[1]
    out["_model"] = model
    out["_dump"] = os.path.basename(dump_dir)
    return out


def measure_seq_frame(path, depth=None, src_gray=None, gt_depth=None, gt_depth_kind="disparity",
                      gt_depth_valid=None,
                      warp_mask=None, warp_mapping=None, warp_mapping_shape=None,
                      src_rgb=None, hdr_scale=None):
    """Spatial metrics for one harness SBS frame and its authenticated sidecars."""
    sbs_rgb = load_rgb(path)
    sbs = rgb_luma(sbs_rgb)
    left, right = split_eyes(sbs)
    left_rgb, right_rgb = split_eyes(sbs_rgb)
    map_left = map_right = None
    out = {"_dump": os.path.basename(path)}
    # Ad-hoc dumps do not have a production inverse-warp sidecar, so retain the permissive
    # cross-eye matcher there. Harness eval replaces it below with exact localized evidence.
    if warp_mapping is None:
        field = disparity_field(left, right)
        if field is not None:
            _, dys, weights = field
            out["vmisalign_p99_px"] = weighted_pct(
                np.abs(dys), weights, VERTICAL_MISALIGNMENT_QUANTILE)
            out["vmisalign_p99_pct"] = (
                out["vmisalign_p99_px"] / left.shape[0] * 100.0)
    if depth is not None:
        if gt_depth is not None:
            gt_metrics = depth_ground_truth_metrics(
                depth, gt_depth, gt_depth_kind, gt_depth_valid)
            if gt_metrics:
                out.update(gt_metrics)
    if src_gray is not None:
        if warp_mapping is not None:
            if warp_mapping_shape is None:
                raise ValueError("exact warp mapping is missing its shape contract")
            if warp_mapping.shape != sbs.shape:
                raise ValueError(
                    f"warp map {warp_mapping.shape} does not match SBS {sbs.shape}")
            map_left, map_right = split_eyes(warp_mapping)
            expected_left = _exact_expected_source_luma(
                left.shape, src_gray, map_left, warp_mapping_shape,
                src_rgb=src_rgb, hdr_scale=hdr_scale)
            expected_right = _exact_expected_source_luma(
                right.shape, src_gray, map_right, warp_mapping_shape,
                src_rgb=src_rgb, hdr_scale=hdr_scale)
            lm = exact_source_relative_metrics(
                left, src_gray, map_left, warp_mapping_shape,
                eye_rgb=left_rgb if src_rgb is not None else None, src_rgb=src_rgb,
                hdr_scale=hdr_scale, expected_evidence=expected_left)
            rm = exact_source_relative_metrics(
                right, src_gray, map_right, warp_mapping_shape,
                eye_rgb=right_rgb if src_rgb is not None else None, src_rgb=src_rgb,
                hdr_scale=hdr_scale, expected_evidence=expected_right)
            vertical_px, vertical_pct, vertical_support = exact_vertical_misalignment(
                left, right, src_gray, map_left, map_right, warp_mapping_shape,
                src_rgb=src_rgb, hdr_scale=hdr_scale,
                expected_evidence=(expected_left, expected_right))
            out["vmisalign_support_pct"] = vertical_support
            if vertical_px is not None:
                out["vmisalign_p99_px"] = vertical_px
                out["vmisalign_p99_pct"] = vertical_pct
        else:
            lm = source_relative_metrics(left, src_gray)
            rm = source_relative_metrics(right, src_gray)
        for key in ("source_coverage_pct", "image_integrity_pct"):
            vals = [m[key] for m in (lm, rm) if key in m]
            if vals:
                out[key] = min(vals)
        for key in ("source_coverage_worst_patch_bad_pct",
                    "image_integrity_worst_patch_bad_pct"):
            vals = [m[key] for m in (lm, rm) if key in m]
            if vals:
                out[key] = max(vals)
        for key in ("source_fidelity_support_pct", "image_integrity_support"):
            supports = [m[key] for m in (lm, rm) if key in m]
            if supports:
                out[key] = min(supports)
    if warp_mapping is not None:
        if warp_mapping_shape is None:
            raise ValueError("exact warp mapping is missing its shape contract")
        coverage_map = None
        if warp_mask is not None:
            hole_map = warp_mask if warp_mask.ndim == 2 else warp_mask[..., 0]
            coverage_map = hole_map < 0.5
        binocular_geometry = _exact_binocular_geometry(
            warp_mapping, warp_mapping_shape, coverage_map)
        if src_gray is not None:
            out.update(exact_visible_disparity_metrics(
                warp_mapping, warp_mapping_shape, src_gray, warp_mask=warp_mask,
                binocular_geometry=binocular_geometry))
            shear = sbs_warp_shear_metrics.measure_cross_row_shear(
                warp_mapping, warp_mapping_shape,
                source=src_rgb if src_rgb is not None else src_gray)
            out.update({key: float(value) for key, value in shear.items()
                        if value is not None})
            if src_rgb is not None:
                perceptual_source = src_rgb
                perceptual_transform = None
                if hdr_scale is not None:
                    perceptual_source = (
                        _srgb_to_linear(src_rgb) * float(hdr_scale)
                    ).astype(np.float16).astype(np.float32)
                    perceptual_transform = _hdr_preview_rgb

                interocular_evidence = (
                    sbs_interocular_phase_chroma.prepare_interocular_evidence(
                        perceptual_source, left_rgb, right_rgb, map_left, map_right,
                        warp_mapping_shape, warp_mask=warp_mask,
                        source_sample_transform=perceptual_transform))
                phase_chroma = (
                    sbs_interocular_phase_chroma
                    .measure_interocular_phase_chroma_prepared(interocular_evidence))
                retained_conflict = {
                    "interocular_phase_orientation_burden_pct",
                    "interocular_phase_orientation_support_pct",
                    "interocular_phase_orientation_support_count",
                    "interocular_phase_orientation_evidence_sufficient",
                }
                out.update({key: float(value) for key, value in phase_chroma.items()
                            if key in retained_conflict and value is not None})

                if hdr_scale is None:
                    photometric = (
                        sbs_interocular_photometric_rivalry
                        .measure_interocular_photometric_rivalry_prepared(
                            interocular_evidence))
                    retained_photometric = {
                        "interocular_exposure_rivalry_burden_pct",
                        "interocular_exposure_rivalry_support_pct",
                        "interocular_exposure_rivalry_support_count",
                        "interocular_exposure_rivalry_evidence_sufficient",
                        "interocular_color_gain_rivalry_burden_pct",
                        "interocular_color_gain_rivalry_support_pct",
                        "interocular_color_gain_rivalry_support_count",
                        "interocular_color_gain_rivalry_evidence_sufficient",
                    }
                    out.update({key: float(value) for key, value in photometric.items()
                                if key in retained_photometric and value is not None})
                else:
                    # The PNG is a diagnostic tone-mapped preview. It cannot authenticate
                    # absolute HDR luminance/gamut rivalry; abstain until raw FP16 eyes plus a
                    # calibrated display transform are saved by the harness.
                    out["interocular_exposure_rivalry_evidence_sufficient"] = 0.0
                    out["interocular_color_gain_rivalry_evidence_sufficient"] = 0.0

                window_depth = None
                if depth is not None:
                    window_depth = resize_depth(
                        depth, int(warp_mapping_shape["source_width"]),
                        int(warp_mapping_shape["source_height"]))
                window = sbs_stereo_window_metrics.measure_stereo_window_violation(
                    warp_mapping, warp_mapping_shape, perceptual_source,
                    depth=window_depth, coverage_mask=coverage_map,
                    source_sample_transform=perceptual_transform, compact=True)
                retained_window = {
                    "experimental_stereo_window_support_count",
                    "experimental_stereo_window_support_pct",
                    "experimental_stereo_window_crossed_burden_pct",
                }
                out.update({key: float(value) for key, value in window.items()
                            if key in retained_window and value is not None})

        out.update(exact_warp_mapping_metrics(
            warp_mapping, warp_mapping_shape, depth=depth, warp_mask=warp_mask,
            binocular_geometry=binocular_geometry))
    return out, sbs, left


SEQUENCE_SPATIAL_WORKERS_ENV = "SBSBENCH_SPATIAL_WORKERS"
SEQUENCE_SPATIAL_BACKEND_ENV = "SBSBENCH_SPATIAL_BACKEND"
SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV = "SBSBENCH_SPATIAL_PIXEL_BUDGET_MPX"
SEQUENCE_SPATIAL_MAX_WORKERS = 24
SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS = 64
SEQUENCE_SPATIAL_DEFAULT_PIXEL_BUDGET_MPX = 24.0
_SEQUENCE_SPATIAL_EXECUTOR = None
_SEQUENCE_SPATIAL_EXECUTOR_BACKEND = None


def _sequence_spatial_worker_count(frame_count, packed_pixels=None):
    """Return a deterministic CPU- and image-memory-bounded spatial worker count."""
    if frame_count < 1:
        return 0
    configured = os.environ.get(SEQUENCE_SPATIAL_WORKERS_ENV)
    if configured is not None:
        try:
            requested = int(configured)
        except ValueError as exc:
            raise ValueError(
                f"{SEQUENCE_SPATIAL_WORKERS_ENV} must be an integer from 1 to "
                f"{SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS}") from exc
        if not 1 <= requested <= SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS:
            raise ValueError(
                f"{SEQUENCE_SPATIAL_WORKERS_ENV} must be from 1 to "
                f"{SEQUENCE_SPATIAL_MAX_CONFIGURED_WORKERS}, got {configured!r}")
        return min(requested, frame_count)
    if frame_count < 8:
        return 1
    workers = min(SEQUENCE_SPATIAL_MAX_WORKERS, os.cpu_count() or 1, frame_count)
    if packed_pixels is not None and packed_pixels > 0:
        try:
            pixel_budget_mpx = float(os.environ.get(
                SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV,
                SEQUENCE_SPATIAL_DEFAULT_PIXEL_BUDGET_MPX))
        except ValueError as exc:
            raise ValueError(
                f"{SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV} must be positive") from exc
        if not np.isfinite(pixel_budget_mpx) or pixel_budget_mpx <= 0.0:
            raise ValueError(f"{SEQUENCE_SPATIAL_PIXEL_BUDGET_ENV} must be positive")
        memory_workers = max(1, int(pixel_budget_mpx * 1_000_000 // packed_pixels))
        workers = min(workers, memory_workers)
    return workers


def _measure_sequence_spatial_job(job):
    """Load and measure one frame in a worker; return scalars only, never image arrays."""
    frame_id = job["frame_id"]
    try:
        depth = load_depth(job["depth_path"]) if job["depth_path"] else None
        src_rgb = load_rgb(job["source_path"]) if job["source_path"] else None
        src = rgb_luma(src_rgb) if src_rgb is not None else None
        gt_depth = load_depth(job["gt_depth_path"]) if job["gt_depth_path"] else None
        gt_valid = None
        if job["gt_valid_path"]:
            with Image.open(job["gt_valid_path"]) as valid_image:
                gt_valid = np.asarray(valid_image.convert("L"), dtype=np.uint8) >= 128
        warp_mask = None
        if job["warp_mask_path"]:
            with Image.open(job["warp_mask_path"]) as mask_image:
                warp_mask = (np.asarray(mask_image.convert("RGB"), np.float32) / 255.0)
        warp_mapping = (
            load_warp_mapping(job["warp_mapping_path"], job["mapping_shape"])
            if job["warp_mapping_path"] else None)
        row, _, _ = measure_seq_frame(
            job["sbs_path"], depth, src, gt_depth, job["gt_kind"],
            gt_depth_valid=gt_valid, warp_mask=warp_mask,
            warp_mapping=warp_mapping, warp_mapping_shape=job["mapping_shape"],
            src_rgb=src_rgb, hdr_scale=job["hdr_scale"])
        row["_frame_id"] = frame_id
        return row
    except Exception as exc:
        raise RuntimeError(
            f"spatial metric worker failed for frame {frame_id}: {exc}") from exc


def enable_reusable_spatial_executor():
    """Keep one ordered worker pool alive across clips in a CLI evaluation process."""
    global _SEQUENCE_SPATIAL_EXECUTOR, _SEQUENCE_SPATIAL_EXECUTOR_BACKEND
    if _SEQUENCE_SPATIAL_EXECUTOR is not None:
        return
    worker_count = _sequence_spatial_worker_count(1_000_000)
    if worker_count <= 1:
        return
    backend = os.environ.get(SEQUENCE_SPATIAL_BACKEND_ENV, "process").strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError(
            f"{SEQUENCE_SPATIAL_BACKEND_ENV} must be 'process' or 'thread', got {backend!r}")
    executor_type = (concurrent.futures.ThreadPoolExecutor if backend == "thread" else
                     concurrent.futures.ProcessPoolExecutor)
    _SEQUENCE_SPATIAL_EXECUTOR = executor_type(max_workers=worker_count)
    _SEQUENCE_SPATIAL_EXECUTOR_BACKEND = backend


def disable_reusable_spatial_executor():
    """Release the optional run-level pool; safe to call repeatedly and from ``atexit``."""
    global _SEQUENCE_SPATIAL_EXECUTOR, _SEQUENCE_SPATIAL_EXECUTOR_BACKEND
    executor, _SEQUENCE_SPATIAL_EXECUTOR = _SEQUENCE_SPATIAL_EXECUTOR, None
    _SEQUENCE_SPATIAL_EXECUTOR_BACKEND = None
    if executor is not None:
        executor.shutdown(wait=True)


atexit.register(disable_reusable_spatial_executor)


def _measure_sequence_spatial_rows(jobs):
    """Run independent jobs serially or in an ordered bounded process pool."""
    packed_pixels = None
    if jobs:
        shape = jobs[0].get("mapping_shape")
        if isinstance(shape, dict):
            width, height = shape.get("width"), shape.get("height")
            if isinstance(width, int) and isinstance(height, int):
                packed_pixels = width * height
    worker_count = _sequence_spatial_worker_count(len(jobs), packed_pixels)
    if worker_count <= 1:
        return [_measure_sequence_spatial_job(job) for job in jobs]
    backend = os.environ.get(SEQUENCE_SPATIAL_BACKEND_ENV, "process").strip().lower()
    if backend not in {"process", "thread"}:
        raise ValueError(
            f"{SEQUENCE_SPATIAL_BACKEND_ENV} must be 'process' or 'thread', got {backend!r}")
    # Direct build_report.py execution selects threads because a spawn-based process executor would
    # recursively execute its top-level report construction. The guarded generate_report.py
    # launcher, evaluator, and rescorer can safely select isolated process workers.
    # A run-level executor removes Windows process-spawn/import cost for every later clip. Submit
    # at most the image-memory-bounded worker count at once even when the shared pool is larger.
    if _SEQUENCE_SPATIAL_EXECUTOR is not None:
        if backend != _SEQUENCE_SPATIAL_EXECUTOR_BACKEND:
            raise ValueError("spatial backend changed while a reusable executor was active")
        rows = []
        for start in range(0, len(jobs), worker_count):
            rows.extend(_SEQUENCE_SPATIAL_EXECUTOR.map(
                _measure_sequence_spatial_job, jobs[start:start + worker_count]))
        return rows
    executor_type = (concurrent.futures.ThreadPoolExecutor if backend == "thread" else
                     concurrent.futures.ProcessPoolExecutor)
    with executor_type(max_workers=worker_count) as executor:
        # executor.map preserves input order. Materialize before temporal work so a worker failure
        # cannot leak a partial sequence result into aggregate/worst-frame selection.
        return list(executor.map(_measure_sequence_spatial_job, jobs))


def measure_sequence(seq_dir, frames_dir=None):
    """Measure one authenticated harness clip under spatial and motion-aware contracts.

    Temporal image changes are never measured as raw frame differences: ordinary source motion
    would dominate them. ``static_jitter`` uses independently static source support, while
    ``flow_temporal`` compensates source motion using authenticated flow before measuring output
    residuals. Authenticated depth and stereo sidecars add GT-only evidence without becoming
    enhanced-pop labels.
    """
    sbs_by_id = indexed_files(os.path.join(seq_dir, "sbs_*.png"), "sbs_")
    if not sbs_by_id:
        return None
    frame_ids = sorted(sbs_by_id)
    src_by_id = indexed_files(os.path.join(frames_dir, "frame_*.*"), "frame_") if frames_dir else {}
    src_by_id = {i: p for i, p in src_by_id.items()
                 if p.lower().endswith((".png", ".jpg", ".jpeg"))}
    if frames_dir and set(src_by_id) != set(frame_ids):
        missing_src = sorted(set(frame_ids) - set(src_by_id))
        missing_sbs = sorted(set(src_by_id) - set(frame_ids))
        raise ValueError(f"source/SBS frame-id mismatch: missing source={missing_src}, missing SBS={missing_sbs}")
    depth_by_id = indexed_files(os.path.join(seq_dir, "depth_*.png"), "depth_")
    if depth_by_id and set(depth_by_id) != set(frame_ids):
        missing_depth = sorted(set(frame_ids) - set(depth_by_id))
        extra_depth = sorted(set(depth_by_id) - set(frame_ids))
        raise ValueError(f"depth/SBS frame-id mismatch: missing depth={missing_depth}, extra depth={extra_depth}")
    mask_by_id = indexed_files(os.path.join(seq_dir, "warp_mask_*.png"), "warp_mask_")
    if mask_by_id and set(mask_by_id) != set(frame_ids):
        missing_mask = sorted(set(frame_ids) - set(mask_by_id))
        extra_mask = sorted(set(mask_by_id) - set(frame_ids))
        raise ValueError(
            f"warp-mask/SBS frame-id mismatch: missing mask={missing_mask}, extra mask={extra_mask}")
    mapping_by_id = indexed_files(os.path.join(seq_dir, "warp_map_*.f32"), "warp_map_")
    mapping_shape = None
    if mapping_by_id:
        missing_mapping = sorted(set(frame_ids) - set(mapping_by_id))
        extra_mapping = sorted(set(mapping_by_id) - set(frame_ids))
        if missing_mapping or extra_mapping:
            raise ValueError(
                "warp-map/SBS frame-id mismatch: "
                f"missing map={missing_mapping}, extra map={extra_mapping}")
        shape_path = os.path.join(seq_dir, "warp_map_shape.json")
        try:
            with open(shape_path, encoding="utf-8") as shape_file:
                mapping_shape = json.load(shape_file)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid warp-map shape contract {shape_path}: {exc}") from exc
    elif os.path.exists(os.path.join(seq_dir, "warp_map_shape.json")):
        raise ValueError("warp-map shape contract exists without frame maps")
    hdr_scale = None
    hdr_stats_path = os.path.join(seq_dir, "hdr_output_stats.json")
    if os.path.exists(hdr_stats_path):
        try:
            with open(hdr_stats_path, encoding="utf-8") as hdr_file:
                hdr_stats = json.load(hdr_file)
            hdr_scale = float(hdr_stats["input_scale"])
            if not np.isfinite(hdr_scale) or hdr_scale <= 0.0:
                raise ValueError("input_scale must be positive and finite")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid HDR preview contract {hdr_stats_path}: {exc}") from exc
    gt_by_id = indexed_files(
        os.path.join(frames_dir, "gt_depth", "frame_*.*"), "frame_") if frames_dir else {}
    if gt_by_id and set(gt_by_id) != set(frame_ids):
        missing_gt = sorted(set(frame_ids) - set(gt_by_id))
        extra_gt = sorted(set(gt_by_id) - set(frame_ids))
        raise ValueError(f"GT-depth/SBS frame-id mismatch: missing GT={missing_gt}, extra GT={extra_gt}")
    gt_valid_by_id = indexed_files(
        os.path.join(frames_dir, "gt_depth_valid", "frame_*.*"),
        "frame_") if frames_dir else {}
    if gt_valid_by_id and set(gt_valid_by_id) != set(gt_by_id):
        missing_valid = sorted(set(gt_by_id) - set(gt_valid_by_id))
        extra_valid = sorted(set(gt_valid_by_id) - set(gt_by_id))
        raise ValueError(
            "GT-depth-valid/frame-id mismatch: "
            f"missing validity={missing_valid}, extra validity={extra_valid}")
    gt_right_by_id = indexed_files(
        os.path.join(frames_dir, "gt_right", "frame_*.*"), "frame_") if frames_dir else {}
    if gt_right_by_id and set(gt_right_by_id) != set(frame_ids):
        missing_gt = sorted(set(frame_ids) - set(gt_right_by_id))
        extra_gt = sorted(set(gt_right_by_id) - set(frame_ids))
        raise ValueError(
            f"GT-right/SBS frame-id mismatch: missing GT={missing_gt}, extra GT={extra_gt}")
    flow_by_id = indexed_files(
        os.path.join(frames_dir, "gt_flow", "frame_*.npz"), "frame_") if frames_dir else {}
    expected_flow_ids = set(frame_ids[1:])
    if flow_by_id and set(flow_by_id) != expected_flow_ids:
        missing_flow = sorted(expected_flow_ids - set(flow_by_id))
        extra_flow = sorted(set(flow_by_id) - expected_flow_ids)
        raise ValueError(f"GT-flow/frame-id mismatch: missing GT={missing_flow}, extra GT={extra_flow}")
    gt_kind = "disparity"
    require_gt_depth = require_gt_flow = reference_stereo_available = False
    if frames_dir:
        meta_path = os.path.join(frames_dir, "meta.json")
        try:
            with open(meta_path, encoding="utf-8") as meta_file:
                clip_meta = json.load(meta_file)
            if not isinstance(clip_meta, dict):
                raise ValueError("metadata root must be an object")
            if "required_gt_stereo" in clip_meta:
                retired = clip_meta.pop("required_gt_stereo")
                if not isinstance(retired, bool):
                    raise ValueError("retired required_gt_stereo must be boolean")
                if ("reference_stereo_available" in clip_meta and
                        clip_meta["reference_stereo_available"] != retired):
                    raise ValueError("conflicting retired/current stereo reference declarations")
                clip_meta["reference_stereo_available"] = retired
            gt_kind = clip_meta.get("gt_depth_kind", gt_kind)
            # Prepared public clips created before schema 5 already carry `dataset`; infer their
            # evidence contract so upgrading the evaluator cannot silently keep the old fail-open
            # behavior. Newly prepared clips store the explicit flags below.
            require_gt_depth = bool(clip_meta.get("required_gt_depth", clip_meta.get("dataset")))
            require_gt_flow = bool(clip_meta.get(
                "required_gt_flow", clip_meta.get("dataset") == "TartanAir V2"))
            reference_stereo_available = bool(
                clip_meta.get("reference_stereo_available", False))
        except OSError as exc:
            raise ValueError(f"cannot read clip metadata {meta_path}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid clip metadata {meta_path}: {exc}") from exc
    if require_gt_depth and not gt_by_id:
        raise ValueError("clip requires GT depth, but no gt_depth sidecars were found")
    if require_gt_depth and gt_kind not in ("metric", "depth") and not gt_valid_by_id:
        raise ValueError(
            "clip requires disparity GT, but no authenticated gt_depth_valid sidecars were found")
    if require_gt_flow and not flow_by_id:
        raise ValueError("clip requires GT optical flow, but no gt_flow sidecars were found")
    if reference_stereo_available and not gt_right_by_id:
        raise ValueError(
            "clip declares diagnostic stereo reference availability, but no gt_right sidecars "
            "were found")

    spatial_jobs = [{
        "frame_id": frame_id,
        "sbs_path": sbs_by_id[frame_id],
        "depth_path": depth_by_id.get(frame_id),
        "source_path": src_by_id.get(frame_id),
        "gt_depth_path": gt_by_id.get(frame_id),
        "gt_valid_path": gt_valid_by_id.get(frame_id),
        "warp_mask_path": mask_by_id.get(frame_id),
        "warp_mapping_path": mapping_by_id.get(frame_id),
        "mapping_shape": mapping_shape,
        "gt_kind": gt_kind,
        "hdr_scale": hdr_scale,
    } for frame_id in frame_ids]
    rows = _measure_sequence_spatial_rows(spatial_jobs)
    measured_ids = [row.get("_frame_id") for row in rows]
    if measured_ids != frame_ids:
        raise RuntimeError(
            f"spatial metric frame order changed: expected={frame_ids}, got={measured_ids}")

    static_jitters = []
    flow_temporals, depth_gt_lags = [], []
    transition_counts = {
        "temporal_expected_transition_count": max(0, len(frame_ids) - 1),
        "source_temporal_transition_count": 0,
        "static_applicable_transition_count": 0,
        "static_measured_transition_count": 0,
        "flow_applicable_transition_count": 0,
        "flow_measured_transition_count": 0,
        "gt_depth_transition_count": 0,
        "gt_depth_lag_measured_transition_count": 0,
    }
    prev_left = prev_right = prev_src = prev_gt_depth = prev_gt_valid = None
    if len(frame_ids) > 1:
        for row, frame_id in zip(rows, frame_ids):
            sbs = rgb_luma(load_rgb(sbs_by_id[frame_id]))
            left, right = split_eyes(sbs)
            # Static and canonical flow-temporal scoring do not consume depth. Decode it in this
            # second pass only when authenticated GT temporal metrics can use it; spatial scoring
            # already measured depth for every frame above.
            depth = (load_depth(depth_by_id[frame_id])
                     if frame_id in depth_by_id and gt_by_id else None)
            src = (rgb_luma(load_rgb(src_by_id[frame_id]))
                   if frame_id in src_by_id else None)
            gt_depth = load_depth(gt_by_id[frame_id]) if frame_id in gt_by_id else None
            gt_valid = None
            if frame_id in gt_valid_by_id:
                with Image.open(gt_valid_by_id[frame_id]) as valid_image:
                    gt_valid = np.asarray(valid_image.convert("L"), dtype=np.uint8) >= 128
            if prev_left is not None:
                if src is not None and prev_src is not None:
                    transition_counts["source_temporal_transition_count"] += 1
                    jitter, support = static_region_jitter(
                        left, right, prev_left, prev_right, src, prev_src)
                    row["static_support"] = support
                    if support >= TEMPORAL_MIN_SUPPORT:
                        transition_counts["static_applicable_transition_count"] += 1
                    if jitter is not None:
                        row["static_jitter"] = jitter
                        static_jitters.append(jitter)
                        transition_counts["static_measured_transition_count"] += 1
                    reference_flow = reference_valid = None
                    if frame_id in flow_by_id:
                        with np.load(flow_by_id[frame_id], allow_pickle=False) as flow_data:
                            reference_flow = np.asarray(flow_data["flow"], dtype=np.float32)
                            if "valid" in flow_data:
                                reference_valid = np.asarray(flow_data["valid"], dtype=bool)
                    # Canonical temporal metrics use authenticated dataset flow only; ordinary
                    # clips abstain and may be inspected by the separately versioned oracle.
                    if reference_flow is not None:
                        flow_temporal, _flow_depth_diagnostic, flow_support = (
                            flow_temporal_metrics(
                                left, right, prev_left, prev_right, src, prev_src,
                                reference_flow=reference_flow,
                                reference_valid=reference_valid))
                        row["flow_support"] = flow_support
                        if flow_support >= TEMPORAL_MIN_SUPPORT:
                            transition_counts["flow_applicable_transition_count"] += 1
                        if flow_temporal is not None:
                            row["flow_temporal"] = flow_temporal
                            flow_temporals.append(flow_temporal)
                            transition_counts["flow_measured_transition_count"] += 1
                if depth is not None and gt_depth is not None and prev_gt_depth is not None:
                    transition_counts["gt_depth_transition_count"] += 1
                    depth_gt_lag = depth_ground_truth_lag(
                        depth, gt_depth, prev_gt_depth, gt_kind,
                        validity=gt_valid, previous_validity=prev_gt_valid)
                    if depth_gt_lag is not None:
                        row["depth_gt_lag_f1"] = depth_gt_lag
                        depth_gt_lags.append(depth_gt_lag)
                        transition_counts["gt_depth_lag_measured_transition_count"] += 1
            prev_left, prev_right = left, right
            prev_src, prev_gt_depth = src, gt_depth
            prev_gt_valid = gt_valid
    agg = aggregate(rows)
    agg.update({key: float(value) for key, value in transition_counts.items()})
    if (transition_counts["static_measured_transition_count"] !=
            transition_counts["static_applicable_transition_count"]):
        raise ValueError(
            "static temporal metric missing on an evidence-qualified transition: "
            f"measured={transition_counts['static_measured_transition_count']}, "
            f"applicable={transition_counts['static_applicable_transition_count']}")
    if (transition_counts["flow_measured_transition_count"] !=
            transition_counts["flow_applicable_transition_count"]):
        raise ValueError(
            "flow temporal metric missing on an evidence-qualified transition: "
            f"measured={transition_counts['flow_measured_transition_count']}, "
            f"applicable={transition_counts['flow_applicable_transition_count']}")
    if static_jitters:
        agg["static_jitter_p50"] = float(np.percentile(static_jitters, 50))
        agg["static_jitter_p95"] = float(np.percentile(static_jitters, 95))
    if flow_temporals:
        agg["flow_temporal_p50"] = float(np.percentile(flow_temporals, 50))
        agg["flow_temporal_p95"] = float(np.percentile(flow_temporals, 95))
    if depth_gt_lags:
        agg["depth_gt_lag_f1_p50"] = float(np.percentile(depth_gt_lags, 50))
        agg["depth_gt_lag_f1_p95"] = float(np.percentile(depth_gt_lags, 95))
    if require_gt_depth:
        missing = [k for k in ("depth_gt_affine_nrmse_pct", "depth_gt_edge_f1")
                   if k not in agg]
        if (transition_counts["gt_depth_lag_measured_transition_count"] !=
                transition_counts["gt_depth_transition_count"]):
            missing.append("depth_gt_lag_f1_p95")
        if missing:
            raise ValueError(f"required GT-depth metrics unavailable: {missing}")
    if require_gt_flow and "flow_temporal_p95" not in agg:
        raise ValueError("required GT-flow temporal metric unavailable")
    return rows, agg


def metric_delta_class(base, new, spec):
    """Classify an A/B movement using the same tolerance contract as the baseline gate."""
    tolerance = max(spec.get("abs_floor", 0.0), abs(base) * spec.get("rel_tol", 0.0))
    improvement = new - base if spec.get("better") == "higher" else base - new
    if improvement > tolerance:
        return "improved"
    if improvement < -tolerance:
        return "regressed"
    return "noise"


def metric_gate_failed(base, new, spec):
    """Whether a committed-baseline gate should fail for this metric role."""
    role = spec.get("role", "diagnostic")
    if role == "diagnostic":
        return False
    if role == "hard":
        if "hard_min" in spec and new < spec["hard_min"]:
            return True
        if "hard_max" in spec and new > spec["hard_max"]:
            return True
        return False
    return metric_delta_class(base, new, spec) == "regressed"


def metric_evidence_state(metric, spec, observed, clip_meta=None):
    """Resolve a metric's independent evidence contract.

    Returns ``applicable``, ``unsupported``, or ``missing``.  Crucially, the metric's own numeric
    value never makes its evidence applicable: doing so would let an unauthenticated GT value opt
    itself into a gate, or let a detector value bypass its missing support measurement.
    """
    clip_meta = clip_meta or {}
    requirement = spec.get("requires", "always")
    # A synthetic sidecar can be useful for temporal boundary lag without being a fair monocular
    # depth-accuracy target. Only clips that explicitly authenticate required GT depth may make
    # global accuracy/polarity metrics applicable. Resolve this from provenance before looking at
    # an opportunistically emitted numeric value, otherwise the value would opt itself into a
    # hard gate.
    if requirement == "gt_depth_accuracy":
        return "applicable" if clip_meta.get("required_gt_depth") is True else "unsupported"
    if requirement in EVIDENCE_SUPPORT_REQUIREMENTS:
        support = observed.get(requirement)
        # Missing/invalid support is not proof of exemption.  Only a measured value below the
        # detector's declared evidence minimum is N/A; enough support requires the metric on that
        # same evidence.
        if (isinstance(support, (bool, np.bool_)) or not np.isscalar(support) or
                not np.issubdtype(np.asarray(support).dtype, np.number) or
                not np.isfinite(support) or float(support) < 0.0):
            return "missing"
        return ("applicable" if float(support) >= EVIDENCE_SUPPORT_REQUIREMENTS[requirement]
                else "unsupported")
    if requirement == "gt_depth":
        return ("applicable" if
                bool(clip_meta.get("gt_depth_kind") or clip_meta.get("required_gt_depth"))
                else "unsupported")
    if requirement == "gt_depth_temporal":
        has_gt = bool(clip_meta.get("gt_depth_kind") or clip_meta.get("required_gt_depth"))
        count = clip_meta.get("source_frame_count")
        return ("applicable" if has_gt and (count is None or count > 1) else "unsupported")
    if requirement == "multi_frame":
        count = clip_meta.get("source_frame_count")
        return "applicable" if count is None or count > 1 else "unsupported"
    if requirement in ("static_support", "flow_support"):
        count = clip_meta.get("source_frame_count")
        if count is not None and count <= 1:
            return "unsupported"
        support = observed.get(requirement)
        # A missing/invalid support measurement on a multi-frame clip is not proof of exemption.
        # Fail closed; only a measured lack of reliable pixels makes the temporal metric N/A.
        if (isinstance(support, (bool, np.bool_)) or not np.isscalar(support) or
                not np.issubdtype(np.asarray(support).dtype, np.number) or
                not np.isfinite(support)):
            return "missing"
        return "applicable" if float(support) >= TEMPORAL_MIN_SUPPORT else "unsupported"
    if requirement == "always":
        return "applicable"
    raise ValueError(f"unknown metric evidence requirement {requirement!r} for {metric}")


def metric_evidence_applicable(metric, spec, observed, clip_meta=None):
    """Whether a metric must be present; missing evidence deliberately fails closed."""
    return metric_evidence_state(metric, spec, observed, clip_meta) != "unsupported"


def metric_value_valid(metrics, metric):
    """True only for finite numeric evidence (booleans are not metric samples)."""
    value = metrics.get(metric)
    return (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
            and np.issubdtype(np.asarray(value).dtype, np.number)
            and bool(np.isfinite(value)))


def frame_label_evidence(row, metric_specs, clip_meta=None):
    """Return a per-frame label vector with explicit validity/support and abstention.

    A model must never learn an invented zero from a detector that had no evidence. Only metrics
    explicitly tagged `label` in thresholds.json are exported. Unsupported evidence is a valid
    N/A state; missing/invalid required evidence makes the frame ineligible for label selection.
    """
    evidence = {}
    missing_required = []
    for metric, spec in metric_specs.items():
        label_role = spec.get("label")
        if label_role is None:
            continue
        requirement = spec.get("requires", "always")
        support = row.get(requirement) if requirement in EVIDENCE_SUPPORT_REQUIREMENTS else None
        requires_support = requirement in EVIDENCE_SUPPORT_REQUIREMENTS
        evidence_state = metric_evidence_state(metric, spec, row, clip_meta)
        support_valid = (not requires_support or
                         (metric_value_valid(row, requirement) and float(support) >= 0.0))
        if evidence_state == "missing":
            state = "missing"
        elif evidence_state == "unsupported":
            state = "unsupported"
        elif metric_value_valid(row, metric):
            state = "valid"
        else:
            state = "missing"
        item = {"role": label_role, "state": state}
        if state == "valid":
            item["value"] = float(row[metric])
        if requirement in EVIDENCE_SUPPORT_REQUIREMENTS:
            item["support_metric"] = requirement
            item["support"] = float(support) if support_valid else None
        evidence[metric] = item
        if state == "missing" and label_role in {"reward", "hard", "risk"}:
            missing_required.append(metric)
    return {"eligible": not missing_required, "missing_required": missing_required,
            "metrics": evidence}


def worst_hard_metric(aggregates, metric, spec, clip_ids):
    """Return the safety-worst finite value and clip; hard constraints are never averaged."""
    values = [(float(aggregates[clip][metric]), clip) for clip in clip_ids
              if clip in aggregates and metric_value_valid(aggregates[clip], metric)]
    if not values:
        return None, None
    choose = min if "hard_min" in spec else max
    return choose(values, key=lambda item: item[0])


def canonical_run_gate(results):
    """Return the authoritative run gate represented by ``results.json``.

    Reports are secondary views over a completed evaluator run.  They must not turn an A/B metric
    win into an exportable candidate when the runner rejected missing evidence, a performance or
    quality regression, or a hard constraint.  Validate both the verdict and its supporting lists
    so a malformed/partially-written result fails closed as well.
    """
    meta = results.get("meta")
    run_kind = meta.get("run_kind") if isinstance(meta, dict) else None
    expected_verdict = {
        "baseline-gated": "pass",
        "baseline-update": "pass",
        "comparison-only": "comparison_only",
    }.get(run_kind)
    blockers = []
    for key, kind in (("hard_failures", "hard"),
                      ("evidence_failures", "evidence"),
                      ("regressions", "baseline_or_perf")):
        values = results.get(key)
        if not isinstance(values, list):
            blockers.append({"kind": "invalid_results", "field": key})
        else:
            blockers.extend({"kind": kind, "detail": value} for value in values)
    verdict = results.get("verdict")
    if expected_verdict is None:
        blockers.append({"kind": "invalid_results", "field": "meta.run_kind",
                         "value": run_kind})
    elif verdict != expected_verdict:
        blockers.append({"kind": "verdict", "expected": expected_verdict,
                         "value": verdict})
    return {"passed": not blockers, "run_kind": run_kind, "verdict": verdict,
            "blockers": blockers}


def gate_ab_decision(decision, control_results, treatment_results):
    """Bind an A/B decision to both inputs' canonical evaluator gates."""
    control_gate = canonical_run_gate(control_results)
    treatment_gate = canonical_run_gate(treatment_results)
    gate = {"passed": control_gate["passed"] and treatment_gate["passed"],
            "control": control_gate, "treatment": treatment_gate}
    result = dict(decision)
    result["ab_verdict"] = decision.get("verdict")
    result["canonical_gate"] = gate
    result["screen_candidate"] = (
        decision.get("verdict") == "screen_candidate" and gate["passed"])
    result["perceptual_qualified_candidate"] = (
        result["screen_candidate"]
        and decision.get("perceptual_qualification") == "qualified")
    if not gate["passed"]:
        result["verdict"] = "reject_run_gate"
    return result


def evaluate_ab_decision(control, treatment, clip_ids, metric_specs, hard_clip_ids=None,
                         clip_meta=None):
    """Evaluate a feature A/B without collapsing perceptual axes into one score.

    `control` and `treatment` map clip id -> aggregate metrics. Metric specs declare one of:
      hard        absolute safety/integrity constraint; `hard_min` and/or `hard_max` bounds it
      primary     user-visible quality axis; improvements and regressions remain explicit
      diagnostic  reported only; cannot accept or reject a feature

    A primary-axis tradeoff is deliberately not auto-resolved. It needs the configured/user
    priority plus visual or headset evidence rather than cancellation inside a scalar score.
    Passing this function is only an automated screen: perceptual metrics explicitly marked
    experimental may vote conservatively, but they can never produce a perceptually qualified
    candidate or a reusable training label.
    """
    hard_failures = []
    missing_evidence = []
    axes = {}
    for clip in hard_clip_ids if hard_clip_ids is not None else clip_ids:
        ca, ta = control.get(clip, {}), treatment.get(clip, {})
        for metric, spec in metric_specs.items():
            if spec.get("role", "diagnostic") != "hard":
                continue
            metadata = (clip_meta or {}).get(clip, {})
            control_state = metric_evidence_state(metric, spec, ca, metadata)
            treatment_state = metric_evidence_state(metric, spec, ta, metadata)
            if control_state == treatment_state == "unsupported":
                continue
            invalid_sides = []
            for side, state, values in (("control", control_state, ca),
                                        ("treatment", treatment_state, ta)):
                if state != "applicable":
                    invalid_sides.append(f"{side}_{state}")
                elif not metric_value_valid(values, metric):
                    invalid_sides.append(side)
            if invalid_sides:
                missing_evidence.append({"clip": clip, "metric": metric,
                                         "missing": "+".join(invalid_sides)})
                continue
            before, after = ca.get(metric), ta.get(metric)
            hard_min = spec.get("hard_min")
            hard_max = spec.get("hard_max")
            if ((hard_min is not None and after < hard_min)
                    or (hard_max is not None and after > hard_max)):
                bounds = {k: v for k, v in (("min", hard_min), ("max", hard_max)) if v is not None}
                hard_failures.append({"clip": clip, "metric": metric,
                                      "value": after, "bounds": bounds,
                                      "scope": spec.get("scope", "perceptual")})
    for clip in clip_ids:
        ca, ta = control.get(clip, {}), treatment.get(clip, {})
        for metric, spec in metric_specs.items():
            role = spec.get("role", "diagnostic")
            scope = spec.get("scope", "perceptual")
            if role == "hard":
                continue
            if role != "primary" or scope == "conformance":
                continue
            metadata = (clip_meta or {}).get(clip, {})
            control_state = metric_evidence_state(metric, spec, ca, metadata)
            treatment_state = metric_evidence_state(metric, spec, ta, metadata)
            if control_state == treatment_state == "unsupported":
                continue
            invalid_sides = []
            for side, state, values in (("control", control_state, ca),
                                        ("treatment", treatment_state, ta)):
                if state != "applicable":
                    invalid_sides.append(f"{side}_{state}")
                elif not metric_value_valid(values, metric):
                    invalid_sides.append(side)
            if invalid_sides:
                missing_evidence.append({"clip": clip, "metric": metric,
                                         "missing": "+".join(invalid_sides)})
                continue
            before, after = ca.get(metric), ta.get(metric)
            movement = metric_delta_class(before, after, spec)
            if movement == "noise":
                continue
            axis = spec.get("axis", "uncategorized")
            bucket = axes.setdefault(axis, {"improved": [], "regressed": []})
            bucket[movement].append({"clip": clip, "metric": metric,
                                     "before": before, "after": after})

    improved = sum(len(v["improved"]) for v in axes.values())
    regressed = sum(len(v["regressed"]) for v in axes.values())
    if missing_evidence:
        verdict = "reject_evidence"
    elif hard_failures:
        verdict = "reject_hard"
    elif improved and regressed:
        verdict = "tradeoff"
    elif regressed:
        verdict = "reject_primary"
    elif improved:
        verdict = "screen_candidate"
    else:
        verdict = "screen_neutral"
    perceptual_specs = {
        metric: spec for metric, spec in metric_specs.items()
        if spec.get("scope") == "perceptual" and spec.get("label") is not None
    }
    experimental_perceptual = sorted(
        metric for metric, spec in perceptual_specs.items()
        if spec.get("label_status") != "qualified")
    perceptual_qualification = (
        "qualified" if perceptual_specs and not experimental_perceptual else "experimental")
    return {"verdict": verdict, "hard_failures": hard_failures,
            "conformance_failures": [item for item in hard_failures
                                     if item.get("scope") == "conformance"],
            "perceptual_bound_failures": [item for item in hard_failures
                                          if item.get("scope") != "conformance"],
            "missing_evidence": missing_evidence, "axes": axes,
            "improved": improved, "regressed": regressed,
            "perceptual_qualification": perceptual_qualification,
            "experimental_perceptual_metrics": experimental_perceptual}


def aggregate(rows):
    # Union of keys across ALL rows: a metric missing from frame 0 (e.g. its depth file failed)
    # must not silently vanish from every aggregate.
    keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            # A one-frame comfort/integrity failure cannot be averaged away by a clean clip.
            agg[k] = float(max(vals) if k in HARD_MAX_AGG else min(vals) if k in HARD_MIN_AGG
                           else np.mean(vals))
    agg["_n"] = len(rows)
    agg["_models"] = sorted({r.get("_model", "") for r in rows})
    return agg


def filter_aggregate_by_evidence(rows, aggregate_metrics, metric_specs, clip_meta=None):
    """Keep the compact policy/evidence vector and exclude unsupported detector outputs.

    Standalone validators may expose sub-components, raw residuals, or retired aliases for visual
    debugging. They are deliberately absent from canonical results and cannot leak into model
    features. A low-support numeric value also must not contaminate a clip mean/max merely because
    another frame has enough support. Aggregate-only temporal percentiles are left intact because
    their producers already apply transition support before publishing them.
    """
    policy_names = set(metric_specs)
    evidence_names = {
        spec.get("requires") for spec in metric_specs.values()
        if isinstance(spec.get("requires"), str)
    }
    retained = policy_names | evidence_names
    filtered = {
        key: value for key, value in aggregate_metrics.items()
        if key in retained or key.startswith("_") or key.endswith("_transition_count")
    }
    for metric, spec in metric_specs.items():
        if not any(metric in row for row in rows):
            continue
        values = [float(row[metric]) for row in rows
                  if metric_evidence_state(metric, spec, row, clip_meta) == "applicable"
                  and metric_value_valid(row, metric)]
        if not values:
            filtered.pop(metric, None)
            continue
        filtered[metric] = float(
            max(values) if metric in HARD_MAX_AGG else
            min(values) if metric in HARD_MIN_AGG else np.mean(values))
    return filtered


# ---------------------------------------------------------------------------- main

FMT = ["vmisalign_p99_pct", "source_coverage_pct", "image_integrity_pct"]
SEQ_FMT = [
    "exact_visible_pop_spread_pct",
    "exact_positive_disparity_pct", "exact_negative_disparity_pct",
    "exact_symmetry_residual_p95_pct", "exact_polarity_ok",
    "source_coverage_pct", "source_coverage_worst_patch_bad_pct",
    "image_integrity_pct", "image_integrity_worst_patch_bad_pct",
    "vmisalign_p99_pct",
    "exact_mapping_stretch_pct", "exact_mapping_fold_pct",
    "warp_cross_row_shear_severity_pct",
    "experimental_stereo_window_crossed_burden_pct",
    "interocular_phase_orientation_burden_pct",
    "interocular_exposure_rivalry_burden_pct",
    "interocular_color_gain_rivalry_burden_pct",
    "static_jitter_p95", "flow_temporal_p95",
    "depth_gt_affine_nrmse_pct", "depth_gt_edge_f1",
]
TEMPORAL_KEYS = [
    "static_jitter_p95", "flow_temporal_p95",
    "depth_gt_lag_f1_p95",
]


def print_table(rows, agg, fmt=FMT):
    hdr = f"{'dump':<26}{'model':<22}" + "".join(f"{k.replace('pop_', '').replace('_px', ''):>13}" for k in fmt)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r.get('_dump', ''):<26}{r.get('_model', ''):<22}"
        line += "".join(f"{r[k]:>13.3f}" if k in r else f"{'-':>13}" for k in fmt)
        print(line)
    print("-" * len(hdr))
    line = f"{'MEAN (n=%d)' % agg['_n']:<48}"
    line += "".join(f"{agg[k]:>13.3f}" if k in agg else f"{'-':>13}" for k in fmt)
    print(line)


def print_diff(agg, base, fmt=FMT):
    print("\nvs baseline:")
    for k in dict.fromkeys((*fmt, *TEMPORAL_KEYS)):
        if k in agg and k in base:
            d = agg[k] - base[k]
            pct = (d / base[k] * 100.0) if base[k] else 0.0
            arrow = "+" if d >= 0 else ""
            print(f"  {k:<16} {base[k]:>10.3f} -> {agg[k]:>10.3f}   {arrow}{d:>8.3f} ({arrow}{pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="Validated visual metrics for host SBS output.")
    ap.add_argument("dumps", nargs="*", help="dump_* folders, or a harness output dir with --seq")
    ap.add_argument("--glob", help="glob pattern for dump folders (quote it)")
    ap.add_argument("--seq", help="harness clip: a directory of sbs_*.png (adds temporal metrics)")
    ap.add_argument(
        "--frames",
        help="harness input frames and optional authenticated GT/temporal sidecars")
    ap.add_argument("--json", help="write the scorecard JSON here")
    ap.add_argument("--baseline", help="print deltas vs this scorecard JSON")
    args = ap.parse_args()

    if args.seq:
        res = measure_sequence(args.seq, args.frames)
        if not res:
            sys.exit(f"no sbs_*.png in {args.seq}")
        rows, agg = res
        print_table(rows, agg, SEQ_FMT)
        if args.baseline:
            with open(args.baseline) as f:
                base = json.load(f)
            print_diff(agg, base.get("aggregate", base), SEQ_FMT)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"aggregate": agg, "frames": rows}, f, indent=2)
            print(f"\nwrote {args.json}")
        return

    dirs = list(args.dumps)
    if args.glob:
        dirs += glob.glob(args.glob)
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        ap.error("no dump folders given (positional args or --glob)")

    rows = []
    for d in sorted(dirs):
        m = measure(d)
        if m:
            rows.append(m)
        else:
            print(f"skip {d}: no sbs.png", file=sys.stderr)
    if not rows:
        sys.exit("no measurable dumps")

    agg = aggregate(rows)
    print_table(rows, agg)

    if args.baseline:
        with open(args.baseline) as f:
            base = json.load(f)
        print_diff(agg, base.get("aggregate", base))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"aggregate": agg, "frames": rows}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
