#!/usr/bin/env python3
"""
sbsbench - validated visual metrics for Apollo's host SBS 3D output.

Runs on real "Dump 3D" output (the actual sbs.png the client receives + the depth.png
that produced it), so the numbers reflect the LIVE pipeline, not a CPU replica -- this is
the whole point vs. warpsim (see docs/sbs-benchmark-plan.md). Each metric is a number that
should move with a real quality change, so improvements can be A/B'd against a baseline.

Metric families:
  pop_px / pop_pct      Horizontal L<->R disparity (the "3D depth" you get). Tile phase
                        correlation between the two eyes; p50/p95 of |dx|. Higher = more pop.
  pop_spread_px/pct     Near-to-far disparity RANGE (weighted p95-p5 of SIGNED dx) = the stereo
                        VOLUME, invariant to where the zero-parallax plane sits. Use this (not
                        pop_px) to judge subject-anchored modes: they recenter the field on the
                        subject, dropping median|dx| without losing depth. Higher = more volume.
  vmisalign_px          Median |dy| between eyes. Should be ~0; nonzero = a geometry fault
                        (eyes must differ by horizontal parallax only).
  depth_spread          p95-p5 of the normalized depth map = pop available at the SOURCE
                        (model + normalization). Separates "flat model" from "flat warp".
  source coverage /     Hard integrity constraints after horizontally aligning each eye to the
  image integrity       source; catches missing, black, or collapsed output regions.
  source halo/stretch   Validated silhouette artifacts relative to aligned source structure.
  GT depth accuracy     Scale/shift-invariant RMSE and boundary F1 on clips with gt_depth sidecars.
  flow temporal         Output/depth residual after exact or classical optical-flow compensation.

Usage:
  python sbsbench.py DUMP_DIR [DUMP_DIR ...]        # one or more dump_* folders
  python sbsbench.py --glob "E:/ApolloDev/sbs_dump/dump_2026070*"   # shell-free globbing
  python sbsbench.py DUMP ... --json out.json       # write the scorecard
  python sbsbench.py DUMP ... --baseline base.json  # print deltas vs a saved scorecard

Dependencies: numpy + Pillow only.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------- io

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def load_gray(path):
    a = load_rgb(path)
    # Rec.709 luma; the depth PNG is grayscale so any channel would do, but this is general.
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


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


def split_eyes(sbs_gray):
    """SBS is [left | right], each half the width."""
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


# ------------------------------------------------------------------- pop / geometry

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
            dxs.append(dx)
            dys.append(dy)
            wts.append(v)
    if not dxs:
        return None
    dxs = np.array(dxs)
    dys = np.array(dys)
    wts = np.array(wts)
    return dxs, dys, wts


def weighted_pct(vals, wts, q):
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    c = np.cumsum(wts)
    c /= c[-1]
    return float(np.interp(q, c, vals))


def pop_spread(dxs, wts):
    """Near-to-far stereo VOLUME: weighted p95 - p5 of the SIGNED horizontal disparity. Unlike
    pop_px (median |dx|), this measures the disparity RANGE independent of where the zero-parallax
    plane sits, so subject anchoring -- which recenters the whole field on the subject (shifting
    the median toward 0) without collapsing the range -- is scored on the depth it actually
    delivers, not penalized for placing the subject at the screen. Also correctly gives ~0 to a
    flat scene shifted bodily forward (high |dx|, no structure), which pop_px wrongly rewards."""
    return weighted_pct(dxs, wts, 0.95) - weighted_pct(dxs, wts, 0.05)


REFERENCE_STREAM_ASPECT = 5120.0 / 2160.0


def perceived_disparity_pct(disparity_px, eye_width, eye_height):
    """Disparity as a reference-aspect-equivalent percentage of image geometry.

    Raw percent-of-width changes meaning when the requested image shape changes. Convert through
    eye height and express the result at the validated 5120x2160 reference aspect. No physical
    display size or placement is assumed.
    """
    width = max(float(eye_width), 1.0)
    height = max(float(eye_height), 1.0)
    aspect_scale = (width / height) / REFERENCE_STREAM_ASPECT
    return float(disparity_px) * 100.0 / width * aspect_scale


def comfort_disparity(dxs, wts, eye_width, eye_height, tail=0.99):
    """Signed disparity tails as reference-aspect-equivalent image percentages.

    Physical crossed/uncrossed naming requires a calibrated display convention and headset FOV,
    which the host PNG does not carry. Keep the two signed sides explicit and gate both; this
    prevents a recentered field from hiding an excessive tail in a single absolute statistic.
    """
    lo = weighted_pct(dxs, wts, 1.0 - tail)
    hi = weighted_pct(dxs, wts, tail)
    return (perceived_disparity_pct(max(0.0, hi), eye_width, eye_height),
            perceived_disparity_pct(max(0.0, -lo), eye_width, eye_height))


# --------------------------------------------------------- disocclusion metrics

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
    return np.asarray(
        Image.fromarray((gray * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
        dtype=np.float32) / 255.0


# Reference eye width the pixel-unit tuning was done at (full-res movie runs). All band/run/reach
# windows scale by (ew / REF_EW) so a metric means the same thing at any output resolution.
REF_EW = 3066.0
# Absolute floor for a "real" silhouette: normalized-depth step per NATIVE depth pixel. Percentile
# thresholds alone always find "edges" (even pure noise on flat content); AND-ing this floor makes
# flat scenes legitimately return zero. Real silhouettes measure ~0.1-0.3/px at depth res.
MIN_DEPTH_STEP = 0.04
MIN_DISOCC_FRAC = 0.001  # ratios below 0.1% eye support are statistically meaningless
HARD_MAX_AGG = {"positive_disparity_pct", "negative_disparity_pct", "vmisalign_pct"}
HARD_MIN_AGG = {"source_coverage_pct", "image_integrity_pct"}


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
    up = Image.fromarray(edge_d.astype(np.uint8) * 255).resize((ew, eh), Image.NEAREST)
    return np.asarray(up) > 127


def silhouette_band(depth, ew, eh, edge_pct=99.3, band_px=28):
    """Silhouette edge mask plus the narrow disocclusion band beside it and a clean reference
    region, all at eye res. band_px is in REF_EW pixels and scales with the eye width."""
    edge = silhouette_edges(depth, ew, eh, edge_pct)
    s = eye_scale(ew)
    band_s = max(4, round(band_px * s))
    excl = max(1, round(2 * s))
    band = hdilate(edge, band_s) & ~hdilate(edge, excl)  # beside the silhouette, not the edge itself
    ref = ~hdilate(edge, band_s + max(2, round(8 * s)))  # undisturbed regions away from silhouettes
    return edge, band, ref


def disocclusion_metrics(eye, depth):
    """From the depth silhouettes, measure the fill quality in the disocclusion band.

    disocc_frac  fraction of eye pixels in a band beside a real silhouette (how much the warp had
                 to invent). disocc_smear  horizontal-detail deficit there: 1 - |d/dx eye|(band) /
                 |d/dx eye|(clean). 0 = fill as crisp as clean regions; ->1 = smeared/stretched."""
    eh, ew = eye.shape
    _, band, ref = silhouette_band(depth, ew, eh)
    frac = float(band.mean())
    if frac < MIN_DISOCC_FRAC:
        return frac, None
    gx = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    b = float(gx[band].mean())
    r = float(gx[ref].mean()) if ref.any() else 0.0
    smear = float(np.clip(1.0 - b / (r + 1e-4), 0.0, 1.0)) if r > 0 else 0.0
    return frac, smear


def hdist_x(src, maxd):
    """Per-pixel horizontal distance (px) to the nearest True in `src`, capped at maxd."""
    dist = np.full(src.shape, maxd, np.float32)
    dist[src] = 0.0
    acc = src.copy()
    for s in range(1, maxd + 1):
        acc = hdilate(acc, 1)
        hit = acc & (dist == maxd)
        dist[hit] = s
    return dist


def _herode(a, r):
    """Horizontal erosion (min over +/-r): a boolean pixel survives iff it is in a True run of
    length >= 2r+1, so it flags pixels inside a long horizontal run."""
    return np.stack([np.roll(a, -o, axis=1) for o in range(-r, r + 1)]).min(0)


def stretch_band(eye, depth, edge_pct=99.0, gthr=0.02, min_run=20, reach=220):
    """Large horizontal DISOCCLUSION STRETCH beside silhouettes -- the background rubber-banded to
    fill the gap the foreground uncovered (eye-asymmetric: left eye smears left, right eye right).
    Unlike disocc_smear (narrow-band detail deficit) this measures the EXTENT of the big smear.

    Signature: a wide horizontal run of LOW horizontal gradient that still has VERTICAL structure
    (a horizontally smeared texture = vertical streaks), sitting within `reach` (REF_EW px) of a
    depth silhouette. A smooth background stretched invisibly (no texture) is correctly not
    flagged. Window/threshold params scale with the eye width (per-px gradients scale inversely
    with resolution, so gthr scales by 1/s).

    Returns stretch_area = fraction of the eye that is stretched fill, in per-mille (x1000)."""
    eh, ew = eye.shape
    s = eye_scale(ew)
    gthr_s = min(0.1, gthr / max(s, 1e-3))
    gx = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    gy = np.abs(np.diff(eye, axis=0, prepend=eye[:1, :]))
    streak = (gx < gthr_s) & (gy > gthr_s)       # smooth in x, structured in y = horizontal smear
    long = _herode(streak, max(3, round(min_run * s / 2)))  # inside a run >= min_run (scaled)
    near = hdilate(silhouette_edges(depth, ew, eh, edge_pct), max(20, round(reach * s)))
    return float((long & near).mean() * 1000.0)


def _hopen(a, r):
    """Horizontal grayscale opening (erode then dilate) with radius r -- removes bright features
    narrower than 2r+1 px, leaving the broad fg/bg. eye - open = a horizontal white top-hat."""
    def shifts(x): return np.stack([np.roll(x, -o, axis=1) for o in range(-r, r + 1)])
    return shifts(shifts(a).min(0)).max(0)


def silhouette_halo(eye, depth, edge_pct=98.5, ridge_r=2, band_px=6):
    """Bright thin FRINGE hugging the silhouette -- the 'white line around the nose': the residual
    bright sliver where the warp/inpaint fill doesn't reach the foreground edge. It is a thin
    bright RIDGE in the eye (brighter than the fg and bg it separates, only a few px wide), sitting
    on the depth silhouette. We detect thin bright ridges with a horizontal white top-hat
    (eye - horizontal-opening, radius ridge_r px) and sample them in the silhouette band. This
    ignores broad bright regions (top-hat ~0) and clean monotonic edges (no ridge).

    Returns (rim_over_p50, rim_over_p95) in luma/255 -- the white-line severity. ridge_r/band_px
    are in REF_EW pixels and scale with eye width; a full-res-thin line becomes sub-pixel at low
    output resolution, where this metric correctly loses sensitivity (use full-res clips for it)."""
    eh, ew = eye.shape
    s = eye_scale(ew)
    edge = silhouette_edges(depth, ew, eh, edge_pct)
    band = hdilate(edge, max(2, round(band_px * s)))  # the rim sits within a few px of the edge
    if not band.any():
        return 0.0, 0.0
    ridge = np.clip(eye - _hopen(eye, max(1, round(ridge_r * s))), 0.0, None)  # white top-hat
    vals = ridge[band] * 255.0
    return float(np.percentile(vals, 50)), float(np.percentile(vals, 95))


def edge_accuracy(depth, src_gray, edge_pct=99.3, col_pct=97.0, maxd=24):
    """Depth-silhouette accuracy vs the true object edge (targets soft/bent/floating silhouettes).

    At depth resolution: distance from each depth silhouette to the nearest strong SOURCE color
    edge (both horizontal edges, since silhouettes are vertical). Small = the depth model's
    silhouette sits on the real object boundary; large = it floats off it. Returns p50/p95 in
    depth-px. Needs the source frame (only computed when --frames is given)."""
    dh, dw = depth.shape
    src = resize_to(src_gray, dw, dh)
    gxd = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    de = gxd >= max(float(np.percentile(gxd, edge_pct)), MIN_DEPTH_STEP)
    if not de.any():
        return 0.0, 0.0
    gxs = np.abs(np.diff(src, axis=1, prepend=src[:, :1]))
    ce = gxs >= np.percentile(gxs, col_pct)
    dist = hdist_x(ce, maxd)[de]
    return float(np.percentile(dist, 50)), float(np.percentile(dist, 95))


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


def source_align_map(eye, src_gray, max_shift=None):
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
    return best, aligned, radius


def source_match_map(eye, src_gray, max_shift=None):
    """Per-pixel source-relative patch error and horizontal search radius."""
    best, _, radius = source_align_map(eye, src_gray, max_shift)
    return best, radius


def source_match_residual(eye, src_gray, max_shift=None):
    """Monocular corruption after allowing intended horizontal stereo displacement.

    For every output patch, find the closest source patch on the same scanline within the normal
    disparity search radius. A clean shifted eye remains near zero; holes, blur, ringing,
    duplicated/stretched texture and other content not explained by horizontal parallax rise.
    Returns p50/p95 in luma/255. Border columns are excluded because no second view exists there.
    """
    best, _, radius = source_align_map(eye, src_gray, max_shift)
    ew = eye.shape[1]
    valid = best[:, radius:ew - radius] if ew > 2 * radius else best
    vals = valid * 255.0
    return float(np.percentile(vals, 50)), float(np.percentile(vals, 95))


def source_relative_metrics(eye, src_gray, depth=None, max_shift=None,
                            coverage_error=24.0 / 255.0):
    """Validated source-relative warp integrity and silhouette artifacts for one eye.

    Horizontal source search makes intended stereo displacement free. Coverage measures how much
    of the interior can still be explained by source content. Integrity measures retention of
    real source texture. Halo and stretch subtract/compare the selected source sample, preventing
    genuine bright outlines or naturally smooth regions from being labeled warp artifacts.
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
    out["image_integrity_pct"] = (float(np.mean(texture_eye[textured] >= 0.25 * texture_src[textured])
                                        * 100.0) if textured.any() else 100.0)

    if depth is None:
        return out
    edge = silhouette_edges(depth, ew, eh, 99.0)
    s = eye_scale(ew)
    halo_band = hdilate(edge, max(2, round(6 * s))) & valid
    if halo_band.any():
        r = max(1, round(2 * s))
        eye_ridge = np.clip(eye - _hopen(eye, r), 0.0, None)
        src_ridge = np.clip(aligned - _hopen(aligned, r), 0.0, None)
        excess = np.clip(eye_ridge - src_ridge, 0.0, None)[halo_band] * 255.0
        out["source_halo_p95"] = float(np.percentile(excess, 95))
    else:
        out["source_halo_p95"] = 0.0

    reach = max(12, round(180 * s))
    near = hdilate(edge, reach) & ~hdilate(edge, max(1, round(2 * s))) & valid
    source_detail = near & (gx_src >= 4.0 / 255.0)
    if source_detail.any():
        collapsed = gx_eye[source_detail] < 0.35 * gx_src[source_detail]
        out["source_stretch_pct"] = float(np.mean(collapsed) * 100.0)
        out["source_stretch_support"] = float(np.mean(source_detail) * 100.0)
    return out


def warp_hole_metrics(left, right, mask_rgb, src_gray=None,
                      coverage_error=24.0 / 255.0):
    """Measure the warp's exact pre-fill holes and whether visible corruption lands there.

    The harness mask contract is R=disocclusion before the active fill and G=still unresolved
    afterward. Hole area itself is context, not quality: stronger valid stereo naturally exposes
    more background. Source-relative residual restricted to that support measures fill fidelity,
    while artifact_in_hole_pct answers the prerequisite question for any future inpainter: what
    fraction of detected corruption is actually inside (or immediately beside) a true hole?
    """
    mask_rgb = np.asarray(mask_rgb, dtype=np.float32)
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] < 2:
        raise ValueError("warp mask must be an RGB image")
    mask_eyes = np.split(mask_rgb, 2, axis=1)
    eyes = (left, right)
    hole_pcts, unresolved_pcts = [], []
    hole_residuals = []
    bad_hole = bad_hole_total = 0
    artifact_in_hole = artifact_total = 0

    for eye, mask in zip(eyes, mask_eyes):
        scale = min(1.0, 256.0 / eye.shape[1])
        ew = max(1, round(eye.shape[1] * scale))
        eh = max(1, round(eye.shape[0] * scale))
        eye_small = resize_to(eye, ew, eh) if scale < 1.0 else eye
        mask_small = np.asarray(
            Image.fromarray((mask[..., :2] * 255.0).astype(np.uint8), mode="LA")
            .resize((ew, eh), Image.NEAREST), dtype=np.uint8) >= 128
        hole, unresolved = mask_small[..., 0], mask_small[..., 1]
        if src_gray is not None:
            best, _, radius = source_align_map(eye_small, src_gray)
        else:
            best, radius = None, max(1, round(40 * eye_scale(ew)))
        valid = np.ones((eh, ew), dtype=bool)
        if ew > 2 * radius:
            valid[:, :radius] = False
            valid[:, ew - radius:] = False
        hole_pcts.append(float(np.mean(hole[valid]) * 100.0))
        unresolved_pcts.append(float(np.mean(unresolved[valid]) * 100.0))
        if src_gray is None:
            continue
        supported_hole = hole & valid
        if supported_hole.any():
            values = best[supported_hole] * 255.0
            hole_residuals.extend(values.tolist())
            bad_hole += int(np.count_nonzero(values > coverage_error * 255.0))
            bad_hole_total += int(values.size)
        artifact = valid & (best > coverage_error)
        if artifact.any():
            # One diagnostic pixel of tolerance covers mask/output rasterization boundaries.
            near_hole = dilate2d(hole | unresolved, 1)
            artifact_in_hole += int(np.count_nonzero(artifact & near_hole))
            artifact_total += int(np.count_nonzero(artifact))

    out = {
        "warp_hole_pct": max(hole_pcts, default=0.0),
        "warp_unresolved_pct": max(unresolved_pcts, default=0.0),
    }
    if src_gray is not None:
        out["hole_source_residual_p95"] = (
            float(np.percentile(hole_residuals, 95)) if hole_residuals else 0.0)
        out["hole_bad_fill_pct"] = (
            100.0 * bad_hole / bad_hole_total if bad_hole_total else 0.0)
        out["artifact_in_hole_pct"] = (
            100.0 * artifact_in_hole / artifact_total if artifact_total else 0.0)
    return out


def resize_depth(depth, w, h):
    """Float depth resize without the 8-bit quantization used by resize_to()."""
    im = Image.fromarray(np.asarray(depth, dtype=np.float32), mode="F")
    return np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32)


def resize_metric_depth(depth, w, h):
    """Resize metric depth without bleeding invalid zero/NaN pixels into valid inverse depth.

    Bilinear interpolation of a valid depth beside zero creates a tiny positive value. Inverting
    that value produces an arbitrarily large false GT error, especially on real RGB-D silhouette
    holes. Interpolate values and validity weights separately, normalize, and accept only pixels
    whose bilinear footprint was effectively all valid.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 1e-6)
    values = resize_depth(np.where(valid, depth, 0.0), w, h)
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
    design = np.column_stack((pv, np.ones_like(pv)))
    scale, shift = np.linalg.lstsq(design, tv, rcond=None)[0]
    if scale <= 0.0:
        return np.full_like(pred, float(np.median(tv))), float(scale)
    return pred * float(scale) + float(shift), float(scale)


def depth_ground_truth_metrics(prediction, ground_truth, kind="disparity"):
    """Scale/shift-invariant relative-depth accuracy plus boundary accuracy.

    Monocular models predict relative disparity, so comparing raw values to metric depth is not
    meaningful. Metric depth is converted to inverse depth, then prediction is affine-aligned on
    valid pixels. Constant-GT scenes use shift-only alignment so hallucinated structure cannot be
    fitted away. The RMSE is normalized by GT robust range (or full normalized range for flat GT).
    """
    pred = np.asarray(prediction, np.float32)
    if kind in ("metric", "depth"):
        gt, valid = resize_metric_depth(ground_truth, pred.shape[1], pred.shape[0])
    else:
        gt = resize_depth(ground_truth, pred.shape[1], pred.shape[0])
        valid = np.isfinite(gt)
    if kind in ("metric", "depth"):
        target = np.zeros_like(gt)
        target[valid] = 1.0 / gt[valid]
    else:
        valid &= gt >= 0.0
        target = gt
    if valid.sum() < 64:
        return None

    tv = target[valid]
    t5, t95 = np.percentile(tv, (5, 95))
    trange = float(t95 - t5)
    if trange < 1e-4:
        aligned, _ = align_relative_depth(pred, target, valid)
        norm = 1.0
    else:
        aligned, _ = align_relative_depth(pred, target, valid)
        norm = trange
    error = aligned[valid] - tv
    si_rmse = float(np.sqrt(np.mean(error * error)) / max(norm, 1e-6) * 100.0)

    gx_t = np.abs(np.diff(target, axis=1, prepend=target[:, :1]))
    gy_t = np.abs(np.diff(target, axis=0, prepend=target[:1, :]))
    gx_p = np.abs(np.diff(aligned, axis=1, prepend=aligned[:, :1]))
    gy_p = np.abs(np.diff(aligned, axis=0, prepend=aligned[:1, :]))
    # A gradient is valid only when both samples used to form it are valid. Otherwise the edge of
    # a missing/zero metric-depth region is incorrectly scored as scene geometry.
    valid_x = valid & np.concatenate((valid[:, :1], valid[:, :-1]), axis=1)
    valid_y = valid & np.concatenate((valid[:1, :], valid[:-1, :]), axis=0)
    edge_threshold = max(0.02, trange * 0.08)
    gt_edge = ((valid_x & (gx_t >= edge_threshold)) |
               (valid_y & (gy_t >= edge_threshold)))
    pred_edge = ((valid_x & (gx_p >= edge_threshold)) |
                 (valid_y & (gy_p >= edge_threshold)))
    if not gt_edge.any():
        edge_f1 = 100.0 if not pred_edge.any() else 0.0
    elif not pred_edge.any():
        edge_f1 = 0.0
    else:
        gt_near = dilate2d(gt_edge, 1)
        pred_near = dilate2d(pred_edge, 1)
        precision = float(np.mean(gt_near[pred_edge]))
        recall = float(np.mean(pred_near[gt_edge]))
        edge_f1 = 200.0 * precision * recall / max(precision + recall, 1e-9)
    return {"depth_gt_si_rmse": si_rmse, "depth_gt_edge_f1": edge_f1}


def static_region_mask(src, prev_src, ew, eh, motion_threshold=3.0 / 255.0):
    """Source-static eye-resolution mask with disparity-radius exclusion around motion."""
    now = resize_to(src, ew, eh)
    before = resize_to(prev_src, ew, eh)
    moving = np.abs(now - before) >= motion_threshold
    radius = max(4, round(40 * eye_scale(ew)))
    return ~hdilate(moving, radius)


def static_region_jitter(left, right, prev_left, prev_right, src, prev_src,
                         motion_threshold=3.0 / 255.0, min_support=0.1):
    """Worst-eye temporal change over source regions that did not move.

    Source-motion pixels are horizontally dilated by the normal disparity radius before exclusion,
    so an output sample that legitimately originated from nearby moving content cannot be mistaken
    for warp jitter. Returns (p95 luma/255, stable support fraction), or (None, support) when too
    little static evidence remains (camera motion / scene cut).
    """
    eh, ew = left.shape
    stable = static_region_mask(src, prev_src, ew, eh, motion_threshold)
    support = float(stable.mean())
    if support < min_support:
        return None, support
    left_delta = np.abs(left - prev_left)[stable] * 255.0
    right_delta = np.abs(right - prev_right)[stable] * 255.0
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
                          depth=None, prev_depth=None, min_support=0.1,
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
    lerr = np.abs(cur_l - prev_l_warp)[mask] * 255.0
    rerr = np.abs(cur_r - prev_r_warp)[mask] * 255.0
    output_p95 = max(float(np.percentile(lerr, 95)), float(np.percentile(rerr, 95)))

    depth_p95 = None
    if depth is not None and prev_depth is not None:
        cur_d = resize_depth(depth, vw, vh)
        old_d = resize_depth(prev_depth, vw, vh)
        prev_d_warp, d_valid = warp_previous_with_flow(old_d, u, v)
        dmask = reliable & d_valid
        if float(dmask.mean()) >= min_support:
            depth_p95 = float(np.percentile(np.abs(cur_d - prev_d_warp)[dmask], 95) * 255.0)
    return output_p95, depth_p95, support


# ----------------------------------------------------------------------- per-frame

def measure(dump_dir):
    sbs_p = os.path.join(dump_dir, "sbs.png")
    depth_p = os.path.join(dump_dir, "depth.png")
    source_p = os.path.join(dump_dir, "source.png")
    if not os.path.exists(sbs_p):
        return None
    sbs = load_gray(sbs_p)
    left, right = split_eyes(sbs)
    ew = left.shape[1]
    eh = left.shape[0]

    out = {}
    field = disparity_field(left, right)
    if field is not None:
        dxs, dys, wts = field
        adx = np.abs(dxs)
        out["pop_px_p50"] = weighted_pct(adx, wts, 0.50)
        out["pop_px_p95"] = weighted_pct(adx, wts, 0.95)
        out["pop_pct_p50"] = perceived_disparity_pct(out["pop_px_p50"], ew, eh)
        out["pop_spread_px"] = pop_spread(dxs, wts)
        out["pop_spread_pct"] = perceived_disparity_pct(out["pop_spread_px"], ew, eh)
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
        out["vmisalign_pct"] = out["vmisalign_px"] / left.shape[0] * 100.0
        out["positive_disparity_pct"], out["negative_disparity_pct"] = comfort_disparity(
            dxs, wts, ew, eh)
        out["tiles"] = int(len(dxs))

    if os.path.exists(depth_p):
        d = load_depth(depth_p)
        out["depth_spread"] = float(np.percentile(d, 95) - np.percentile(d, 5))
        out["disocc_frac"], smear = disocclusion_metrics(left, d)
        if smear is not None:
            out["disocc_smear"] = smear
        out["stretch_area"] = stretch_band(left, d)
        out["rim_over_p50"], out["rim_over_p95"] = silhouette_halo(left, d)
    if os.path.exists(source_p):
        src = load_gray(source_p)
        d = load_depth(depth_p) if os.path.exists(depth_p) else None
        lm, rm = source_relative_metrics(left, src, d), source_relative_metrics(right, src, d)
        for key in ("source_residual_p50", "source_residual_p95", "source_halo_p95",
                    "source_stretch_pct"):
            vals = [m[key] for m in (lm, rm) if key in m]
            if vals:
                out[key] = max(vals)
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
                      warp_mask=None):
    """Spatial metrics for one harness SBS frame. depth (0-1 array) adds disocclusion; the source
    frame (src_gray, 0-1) adds depth-silhouette edge accuracy."""
    sbs = load_gray(path)
    left, right = split_eyes(sbs)
    ew = left.shape[1]
    eh = left.shape[0]
    out = {"_dump": os.path.basename(path)}
    field = disparity_field(left, right)
    if field is not None:
        dxs, dys, wts = field
        adx = np.abs(dxs)
        out["pop_px_p50"] = weighted_pct(adx, wts, 0.50)
        out["pop_px_p95"] = weighted_pct(adx, wts, 0.95)
        out["pop_pct_p50"] = perceived_disparity_pct(out["pop_px_p50"], ew, eh)
        out["pop_spread_px"] = pop_spread(dxs, wts)
        out["pop_spread_pct"] = perceived_disparity_pct(out["pop_spread_px"], ew, eh)
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
        out["vmisalign_pct"] = out["vmisalign_px"] / left.shape[0] * 100.0
        out["positive_disparity_pct"], out["negative_disparity_pct"] = comfort_disparity(
            dxs, wts, ew, eh)
    if depth is not None:
        out["depth_spread"] = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        out["disocc_frac"], smear = disocclusion_metrics(left, depth)
        if smear is not None:
            out["disocc_smear"] = smear
        out["stretch_area"] = stretch_band(left, depth)
        out["rim_over_p50"], out["rim_over_p95"] = silhouette_halo(left, depth)
        if src_gray is not None:
            out["edge_acc_p50"], out["edge_acc_p95"] = edge_accuracy(depth, src_gray)
        if gt_depth is not None:
            gt_metrics = depth_ground_truth_metrics(depth, gt_depth, gt_depth_kind)
            if gt_metrics:
                out.update(gt_metrics)
    if src_gray is not None:
        lm = source_relative_metrics(left, src_gray, depth)
        rm = source_relative_metrics(right, src_gray, depth)
        for key in ("source_residual_p50", "source_residual_p95", "source_halo_p95",
                    "source_stretch_pct"):
            vals = [m[key] for m in (lm, rm) if key in m]
            if vals:
                out[key] = max(vals)
        for key in ("source_coverage_pct", "image_integrity_pct"):
            out[key] = min(lm[key], rm[key])
        supports = [m.get("source_stretch_support") for m in (lm, rm)
                    if m.get("source_stretch_support") is not None]
        if supports:
            out["source_stretch_support"] = min(supports)
    if warp_mask is not None:
        out.update(warp_hole_metrics(left, right, warp_mask, src_gray))
    return out, sbs, left


def measure_sequence(seq_dir, frames_dir=None, expected_flat=False):
    """A harness clip: sbs_*.png (+ depth_*.png) in order. Per-frame spatial metrics plus the
    temporal metrics that target the current pipeline's failure modes:

      flicker         frame-to-frame mean|delta| of the whole SBS luma (x255).
      flicker_disocc  same, but restricted to the disocclusion bands -- isolates inpaint/stretch
                      re-hallucination shimmer from ordinary motion (the ~1/4-res inpaint problem).
      swim            frame-to-frame |depth change| where the SOURCE is static (needs --frames) --
                      the scene-cut / flat-content depth instability, separated from real motion.
      static_jitter   worst-eye p95 output change where a disparity-dilated SOURCE neighborhood
                      stayed static; rejects warp/depth shimmer without counting normal motion.

    On the SAME clip the real motion is identical, so these DELTAS vs baseline are pure changes."""
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
    gt_by_id = indexed_files(
        os.path.join(frames_dir, "gt_depth", "frame_*.*"), "frame_") if frames_dir else {}
    if gt_by_id and set(gt_by_id) != set(frame_ids):
        missing_gt = sorted(set(frame_ids) - set(gt_by_id))
        extra_gt = sorted(set(gt_by_id) - set(frame_ids))
        raise ValueError(f"GT-depth/SBS frame-id mismatch: missing GT={missing_gt}, extra GT={extra_gt}")
    flow_by_id = indexed_files(
        os.path.join(frames_dir, "gt_flow", "frame_*.npz"), "frame_") if frames_dir else {}
    expected_flow_ids = set(frame_ids[1:])
    if flow_by_id and set(flow_by_id) != expected_flow_ids:
        missing_flow = sorted(expected_flow_ids - set(flow_by_id))
        extra_flow = sorted(set(flow_by_id) - expected_flow_ids)
        raise ValueError(f"GT-flow/frame-id mismatch: missing GT={missing_flow}, extra GT={extra_flow}")
    gt_kind = "disparity"
    require_gt_depth = require_gt_flow = False
    if frames_dir:
        meta_path = os.path.join(frames_dir, "meta.json")
        try:
            with open(meta_path, encoding="utf-8") as meta_file:
                clip_meta = json.load(meta_file)
            gt_kind = clip_meta.get("gt_depth_kind", gt_kind)
            # Prepared public clips created before schema 5 already carry `dataset`; infer their
            # evidence contract so upgrading the evaluator cannot silently keep the old fail-open
            # behavior. Newly prepared clips store the explicit flags below.
            require_gt_depth = bool(clip_meta.get("required_gt_depth", clip_meta.get("dataset")))
            require_gt_flow = bool(clip_meta.get(
                "required_gt_flow", clip_meta.get("dataset") == "TartanAir V2"))
        except (OSError, ValueError):
            pass
    if require_gt_depth and not gt_by_id:
        raise ValueError("clip requires GT depth, but no gt_depth sidecars were found")
    if require_gt_flow and not flow_by_id:
        raise ValueError("clip requires GT optical flow, but no gt_flow sidecars were found")
    rows, flicks, bflicks, swims, static_jitters = [], [], [], [], []
    flow_temporals, flow_depths = [], []
    prev_sbs = prev_left = prev_right = prev_depth = prev_src = None
    for frame_id in frame_ids:
        p = sbs_by_id[frame_id]
        depth = load_depth(depth_by_id[frame_id]) if frame_id in depth_by_id else None
        src = load_gray(src_by_id[frame_id]) if frame_id in src_by_id else None
        gt_depth = load_depth(gt_by_id[frame_id]) if frame_id in gt_by_id else None
        warp_mask = (np.asarray(Image.open(mask_by_id[frame_id]).convert("RGB"), np.float32)
                     / 255.0 if frame_id in mask_by_id else None)
        row, sbs, left = measure_seq_frame(
            p, depth, src, gt_depth, gt_kind, warp_mask=warp_mask)
        _, right = split_eyes(sbs)
        row["_frame_id"] = frame_id
        if prev_sbs is not None:
            row["flicker"] = float(np.mean(np.abs(sbs - prev_sbs)) * 255.0)
            flicks.append(row["flicker"])
            if depth is not None:
                _, band, _ = silhouette_band(depth, left.shape[1], left.shape[0])
                if float(band.mean()) >= MIN_DISOCC_FRAC:
                    row["flicker_disocc"] = float(np.mean(np.abs(left - prev_left)[band]) * 255.0)
                    bflicks.append(row["flicker_disocc"])
            if depth is not None and prev_depth is not None and src is not None and prev_src is not None:
                dh, dw = depth.shape
                a = resize_to(src, dw, dh)
                b = resize_to(prev_src, dw, dh)
                static = np.abs(a - b) < (3.0 / 255.0)  # source pixels that didn't move
                if static.any():
                    row["swim"] = float(np.median(np.abs(depth - prev_depth)[static]) * 255.0)
                    swims.append(row["swim"])
            if src is not None and prev_src is not None:
                jitter, support = static_region_jitter(
                    left, right, prev_left, prev_right, src, prev_src)
                row["static_support"] = support
                if jitter is not None:
                    row["static_jitter"] = jitter
                    static_jitters.append(jitter)
                reference_flow = reference_valid = None
                if frame_id in flow_by_id:
                    with np.load(flow_by_id[frame_id], allow_pickle=False) as flow_data:
                        reference_flow = np.asarray(flow_data["flow"], dtype=np.float32)
                        if "valid" in flow_data:
                            reference_valid = np.asarray(flow_data["valid"], dtype=bool)
                flow_temporal, flow_depth, flow_support = flow_temporal_metrics(
                    left, right, prev_left, prev_right, src, prev_src, depth, prev_depth,
                    reference_flow=reference_flow, reference_valid=reference_valid)
                row["flow_support"] = flow_support
                if flow_temporal is not None:
                    row["flow_temporal"] = flow_temporal
                    flow_temporals.append(flow_temporal)
                if flow_depth is not None:
                    row["flow_depth"] = flow_depth
                    flow_depths.append(flow_depth)
        rows.append(row)
        prev_sbs, prev_left, prev_right, prev_depth, prev_src = sbs, left, right, depth, src
    agg = aggregate(rows)
    for name, vals in [("flicker", flicks), ("flicker_disocc", bflicks), ("swim", swims)]:
        if vals:
            agg[name + "_p50"] = float(np.percentile(vals, 50))
            agg[name + "_p95"] = float(np.percentile(vals, 95))
    if static_jitters:
        agg["static_jitter_p50"] = float(np.percentile(static_jitters, 50))
        agg["static_jitter_p95"] = float(np.percentile(static_jitters, 95))
    for name, vals in (("flow_temporal", flow_temporals), ("flow_depth", flow_depths)):
        if vals:
            agg[name + "_p50"] = float(np.percentile(vals, 50))
            agg[name + "_p95"] = float(np.percentile(vals, 95))
    agg.update(sbs_score(agg, expected_flat=expected_flat))
    if require_gt_depth:
        missing = [k for k in ("depth_gt_si_rmse", "depth_gt_edge_f1") if k not in agg]
        if missing:
            raise ValueError(f"required GT-depth metrics unavailable: {missing}")
    if require_gt_flow and "flow_temporal_p95" not in agg:
        raise ValueError("required GT-flow temporal metric unavailable")
    return rows, agg


def _load_score_cfg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")
    try:
        return json.load(open(p))["score"]
    except Exception:
        return {"penalties": {}, "depth": {"metric": "pop_pct_p50", "target": 0.6, "weight": 0.2}}


SCORE_CFG = _load_score_cfg()


def sbs_score(agg, expected_flat=False):
    """Overall 0-100 artifact quality from an aggregate metric dict (see thresholds.json 'score').
    score measures artifact cleanliness only. q_depth is reported separately because stereo
    volume is a required constraint, not quality points that may cancel an artifact regression."""
    pen = 0.0
    for k, spec in SCORE_CFG.get("penalties", {}).items():
        v = agg.get(k)
        if v is None:
            continue
        pen += spec["weight"] * min(v / spec["scale"], 1.0) if spec["scale"] else 0.0
    q_clean = max(0.0, 100.0 - pen)
    d = SCORE_CFG.get("depth", {})
    pop = agg.get(d.get("metric", "pop_pct_p50"), 0.0)
    target = d.get("expected_flat_target", 0.1) if expected_flat else d.get("target", 0.6)
    realized = min(pop / target, 1.0) if target else 0.0
    q_depth = 100.0 * (1.0 - realized) if expected_flat else 100.0 * realized
    return {"q_clean": round(q_clean, 1), "q_depth": round(q_depth, 1),
            "score": round(q_clean, 1)}


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


def evaluate_ab_decision(control, treatment, clip_ids, metric_specs, hard_clip_ids=None):
    """Evaluate a feature A/B without collapsing perceptual axes into one score.

    `control` and `treatment` map clip id -> aggregate metrics. Metric specs declare one of:
      hard        absolute safety/integrity constraint; `hard_min` and/or `hard_max` bounds it
      primary     user-visible quality axis; improvements and regressions remain explicit
      diagnostic  reported only; cannot accept or reject a feature

    A primary-axis tradeoff is deliberately not auto-resolved. It needs the configured/user
    priority plus visual or headset evidence rather than cancellation inside a scalar score.
    """
    hard_failures = []
    axes = {}
    for clip in hard_clip_ids if hard_clip_ids is not None else clip_ids:
        ca, ta = control.get(clip, {}), treatment.get(clip, {})
        for metric, spec in metric_specs.items():
            if spec.get("role", "diagnostic") != "hard":
                continue
            before, after = ca.get(metric), ta.get(metric)
            if before is None or after is None:
                continue
            hard_min = spec.get("hard_min")
            hard_max = spec.get("hard_max")
            if ((hard_min is not None and after < hard_min)
                    or (hard_max is not None and after > hard_max)):
                bounds = {k: v for k, v in (("min", hard_min), ("max", hard_max)) if v is not None}
                hard_failures.append({"clip": clip, "metric": metric,
                                      "value": after, "bounds": bounds})
    for clip in clip_ids:
        ca, ta = control.get(clip, {}), treatment.get(clip, {})
        for metric, spec in metric_specs.items():
            role = spec.get("role", "diagnostic")
            before, after = ca.get(metric), ta.get(metric)
            if before is None or after is None:
                continue
            if role == "hard":
                continue
            if role != "primary":
                continue
            movement = metric_delta_class(before, after, spec)
            if movement == "noise":
                continue
            axis = spec.get("axis", "uncategorized")
            bucket = axes.setdefault(axis, {"improved": [], "regressed": []})
            bucket[movement].append({"clip": clip, "metric": metric,
                                     "before": before, "after": after})

    improved = sum(len(v["improved"]) for v in axes.values())
    regressed = sum(len(v["regressed"]) for v in axes.values())
    if hard_failures:
        verdict = "reject_hard"
    elif improved and regressed:
        verdict = "tradeoff"
    elif regressed:
        verdict = "reject_primary"
    elif improved:
        verdict = "candidate"
    else:
        verdict = "neutral"
    return {"verdict": verdict, "hard_failures": hard_failures, "axes": axes,
            "improved": improved, "regressed": regressed}


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


# ---------------------------------------------------------------------------- main

FMT = ["pop_spread_px", "positive_disparity_pct", "negative_disparity_pct", "vmisalign_px",
       "source_coverage_pct", "image_integrity_pct", "source_halo_p95", "source_stretch_pct"]
SEQ_FMT = ["pop_spread_px", "source_residual_p95", "source_halo_p95", "source_stretch_pct",
           "static_jitter_p95", "flow_temporal_p95", "depth_gt_si_rmse", "depth_gt_edge_f1"]
TEMPORAL_KEYS = ["flicker_p50", "flicker_p95", "flicker_disocc_p50", "flicker_disocc_p95",
                 "swim_p50", "swim_p95", "static_jitter_p50", "static_jitter_p95",
                 "flow_temporal_p50", "flow_temporal_p95", "flow_depth_p50", "flow_depth_p95"]


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
    if "flicker_p50" in agg:
        def t(name, key):
            return f"{name} p50={agg[key + '_p50']:.2f} p95={agg[key + '_p95']:.2f}" if key + "_p50" in agg else ""
        parts = [t("flicker", "flicker"), t("disocc", "flicker_disocc"), t("swim", "swim")]
        print(f"{'temporal (x255)':<48}" + "   ".join(p for p in parts if p))


def print_diff(agg, base, fmt=FMT):
    print("\nvs baseline:")
    for k in list(fmt) + TEMPORAL_KEYS:
        if k in agg and k in base:
            d = agg[k] - base[k]
            pct = (d / base[k] * 100.0) if base[k] else 0.0
            arrow = "+" if d >= 0 else ""
            print(f"  {k:<16} {base[k]:>10.3f} -> {agg[k]:>10.3f}   {arrow}{d:>8.3f} ({arrow}{pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="No-reference visual metrics for host SBS dumps.")
    ap.add_argument("dumps", nargs="*", help="dump_* folders, or a harness output dir with --seq")
    ap.add_argument("--glob", help="glob pattern for dump folders (quote it)")
    ap.add_argument("--seq", help="harness clip: a directory of sbs_*.png (adds temporal metrics)")
    ap.add_argument("--frames", help="the harness INPUT frames dir (enables swim + edge accuracy)")
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
