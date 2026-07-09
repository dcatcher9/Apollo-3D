#!/usr/bin/env python3
"""
sbsbench - no-reference visual metrics for Apollo's host SBS 3D output.

Runs on real "Dump 3D" output (the actual sbs.png the client receives + the depth.png
that produced it), so the numbers reflect the LIVE pipeline, not a CPU replica -- this is
the whole point vs. warpsim (see docs/sbs-benchmark-plan.md). Each metric is a number that
should move with a real quality change, so improvements can be A/B'd against a baseline.

Metrics (single frame; spatial only in this version -- temporal flicker needs a burst dump):
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
  stretch_band          EXPERIMENTAL proxy for the disocclusion stretch/smear beside
                        silhouettes: excess horizontal-smoothness in the band next to strong
                        depth edges vs. the frame as a whole. Higher = more visible band.

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
    """Depth map -> float 0-1. Handles the harness's 16-bit gray PNG (full swim precision), plus
    8-bit gray and the live dump's RGB depth.png."""
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
    for y in range(0, h - tile, stride):
        for x in range(0, w - tile, stride):
            lt = left[y:y + tile, x:x + tile]
            rt = right[y:y + tile, x:x + tile]
            v = float(lt.var())
            if v < min_var:
                continue
            dy, dx = phase_shift(lt, rt)
            # A shift near the unambiguous range edge is unreliable; drop it.
            if abs(dx) >= tile // 2 - 1 or abs(dy) >= tile // 2 - 1:
                continue
            dxs.append(dx); dys.append(dy); wts.append(v)
    if not dxs:
        return None
    dxs = np.array(dxs); dys = np.array(dys); wts = np.array(wts)
    return dxs, dys, wts


def weighted_pct(vals, wts, q):
    order = np.argsort(vals)
    vals = vals[order]; wts = wts[order]
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


# --------------------------------------------------------- disocclusion metrics

def hdilate(mask, px):
    """Dilate a boolean mask horizontally by +/- px columns."""
    out = mask.copy()
    for s in range(1, px + 1):
        out[:, s:] |= mask[:, :-s]
        out[:, :-s] |= mask[:, s:]
    return out


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
    if not band.any():
        return 0.0, 0.0
    gx = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    b = float(gx[band].mean())
    r = float(gx[ref].mean()) if ref.any() else 0.0
    smear = float(np.clip(1.0 - b / (r + 1e-4), 0.0, 1.0)) if r > 0 else 0.0
    return float(band.mean()), smear


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
    shifts = lambda x: np.stack([np.roll(x, -o, axis=1) for o in range(-r, r + 1)])
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


# ----------------------------------------------------------------------- per-frame

def measure(dump_dir):
    sbs_p = os.path.join(dump_dir, "sbs.png")
    depth_p = os.path.join(dump_dir, "depth.png")
    if not os.path.exists(sbs_p):
        return None
    sbs = load_gray(sbs_p)
    left, right = split_eyes(sbs)
    ew = left.shape[1]

    out = {}
    field = disparity_field(left, right)
    if field is not None:
        dxs, dys, wts = field
        adx = np.abs(dxs)
        out["pop_px_p50"] = weighted_pct(adx, wts, 0.50)
        out["pop_px_p95"] = weighted_pct(adx, wts, 0.95)
        out["pop_pct_p50"] = out["pop_px_p50"] / ew * 100.0
        out["pop_spread_px"] = pop_spread(dxs, wts)
        out["pop_spread_pct"] = out["pop_spread_px"] / ew * 100.0
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
        out["tiles"] = int(len(dxs))

    if os.path.exists(depth_p):
        d = load_depth(depth_p)
        out["depth_spread"] = float(np.percentile(d, 95) - np.percentile(d, 5))
        out["disocc_frac"], out["disocc_smear"] = disocclusion_metrics(left, d)
        out["stretch_area"] = stretch_band(left, d)
        out["rim_over_p50"], out["rim_over_p95"] = silhouette_halo(left, d)

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


def measure_seq_frame(path, depth=None, src_gray=None):
    """Spatial metrics for one harness SBS frame. depth (0-1 array) adds disocclusion; the source
    frame (src_gray, 0-1) adds depth-silhouette edge accuracy."""
    sbs = load_gray(path)
    left, right = split_eyes(sbs)
    ew = left.shape[1]
    out = {"_dump": os.path.basename(path)}
    field = disparity_field(left, right)
    if field is not None:
        dxs, dys, wts = field
        adx = np.abs(dxs)
        out["pop_px_p50"] = weighted_pct(adx, wts, 0.50)
        out["pop_px_p95"] = weighted_pct(adx, wts, 0.95)
        out["pop_pct_p50"] = out["pop_px_p50"] / ew * 100.0
        out["pop_spread_px"] = pop_spread(dxs, wts)
        out["pop_spread_pct"] = out["pop_spread_px"] / ew * 100.0
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
    if depth is not None:
        out["depth_spread"] = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        out["disocc_frac"], out["disocc_smear"] = disocclusion_metrics(left, depth)
        out["stretch_area"] = stretch_band(left, depth)
        out["rim_over_p50"], out["rim_over_p95"] = silhouette_halo(left, depth)
        if src_gray is not None:
            out["edge_acc_p50"], out["edge_acc_p95"] = edge_accuracy(depth, src_gray)
    return out, sbs, left


def measure_sequence(seq_dir, frames_dir=None):
    """A harness clip: sbs_*.png (+ depth_*.png) in order. Per-frame spatial metrics plus the
    temporal metrics that target the current pipeline's failure modes:

      flicker         frame-to-frame mean|delta| of the whole SBS luma (x255).
      flicker_disocc  same, but restricted to the disocclusion bands -- isolates inpaint/stretch
                      re-hallucination shimmer from ordinary motion (the ~1/4-res inpaint problem).
      swim            frame-to-frame |depth change| where the SOURCE is static (needs --frames) --
                      the scene-cut / flat-content depth instability, separated from real motion.

    On the SAME clip the real motion is identical, so these DELTAS vs baseline are pure changes."""
    paths = sorted(glob.glob(os.path.join(seq_dir, "sbs_*.png")))
    if not paths:
        return None
    srcs = sorted(glob.glob(os.path.join(frames_dir, "frame_*.*"))) if frames_dir else []
    srcs = [s for s in srcs if s.lower().endswith((".png", ".jpg", ".jpeg"))]
    rows, flicks, bflicks, swims = [], [], [], []
    prev_sbs = prev_left = prev_depth = prev_src = None
    for i, p in enumerate(paths):
        # Replace only the basename (the containing dir may legitimately contain "sbs_").
        depth_p = os.path.join(os.path.dirname(p), os.path.basename(p).replace("sbs_", "depth_"))
        depth = load_depth(depth_p) if os.path.exists(depth_p) else None
        src = load_gray(srcs[i]) if i < len(srcs) else None
        row, sbs, left = measure_seq_frame(p, depth, src)
        if prev_sbs is not None:
            row["flicker"] = float(np.mean(np.abs(sbs - prev_sbs)) * 255.0)
            flicks.append(row["flicker"])
            if depth is not None:
                _, band, _ = silhouette_band(depth, left.shape[1], left.shape[0])
                if band.any():
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
        rows.append(row)
        prev_sbs, prev_left, prev_depth, prev_src = sbs, left, depth, src
    agg = aggregate(rows)
    for name, vals in [("flicker", flicks), ("flicker_disocc", bflicks), ("swim", swims)]:
        if vals:
            agg[name + "_p50"] = float(np.percentile(vals, 50))
            agg[name + "_p95"] = float(np.percentile(vals, 95))
    agg.update(sbs_score(agg))  # overall quality score from the assembled metrics
    return rows, agg


def _load_score_cfg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json")
    try:
        return json.load(open(p))["score"]
    except Exception:
        return {"penalties": {}, "depth": {"metric": "pop_pct_p50", "target": 0.6, "weight": 0.2}}


SCORE_CFG = _load_score_cfg()


def sbs_score(agg):
    """Overall 0-100 SBS quality from an aggregate metric dict (see thresholds.json 'score').
    q_clean penalizes artifacts; q_depth rewards realized stereo; score blends them."""
    pen = 0.0
    for k, spec in SCORE_CFG.get("penalties", {}).items():
        v = agg.get(k)
        if v is None:
            continue
        pen += spec["weight"] * min(v / spec["scale"], 1.0) if spec["scale"] else 0.0
    q_clean = max(0.0, 100.0 - pen)
    d = SCORE_CFG.get("depth", {})
    pop = agg.get(d.get("metric", "pop_pct_p50"), 0.0)
    q_depth = 100.0 * min(pop / d.get("target", 0.6), 1.0) if d.get("target") else 0.0
    dw = d.get("weight", 0.2)
    score = (1.0 - dw) * q_clean + dw * q_depth
    return {"q_clean": round(q_clean, 1), "q_depth": round(q_depth, 1), "score": round(score, 1)}


def aggregate(rows):
    # Union of keys across ALL rows: a metric missing from frame 0 (e.g. its depth file failed)
    # must not silently vanish from every aggregate.
    keys = sorted({k for r in rows for k in r if not k.startswith("_")})
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
    agg["_n"] = len(rows)
    agg["_models"] = sorted({r.get("_model", "") for r in rows})
    return agg


# ---------------------------------------------------------------------------- main

FMT = ["pop_px_p50", "pop_spread_px", "vmisalign_px", "depth_spread",
       "disocc_smear", "stretch_area", "rim_over_p95"]
SEQ_FMT = ["pop_px_p50", "pop_spread_px", "vmisalign_px", "stretch_area", "rim_over_p95",
           "edge_acc_p50", "flicker"]
TEMPORAL_KEYS = ["flicker_p50", "flicker_p95", "flicker_disocc_p50", "flicker_disocc_p95",
                 "swim_p50", "swim_p95"]


def print_table(rows, agg, fmt=FMT):
    hdr = f"{'dump':<26}{'model':<22}" + "".join(f"{k.replace('pop_','').replace('_px',''):>13}" for k in fmt)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r.get('_dump',''):<26}{r.get('_model',''):<22}"
        line += "".join(f"{r[k]:>13.3f}" if k in r else f"{'-':>13}" for k in fmt)
        print(line)
    print("-" * len(hdr))
    line = f"{'MEAN (n=%d)' % agg['_n']:<48}"
    line += "".join(f"{agg[k]:>13.3f}" if k in agg else f"{'-':>13}" for k in fmt)
    print(line)
    if "flicker_p50" in agg:
        def t(name, key):
            return f"{name} p50={agg[key+'_p50']:.2f} p95={agg[key+'_p95']:.2f}" if key + "_p50" in agg else ""
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
