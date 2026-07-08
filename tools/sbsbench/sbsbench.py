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


def split_eyes(sbs_gray):
    """SBS is [left | right], each half the width."""
    w = sbs_gray.shape[1] // 2
    return sbs_gray[:, :w], sbs_gray[:, w:2 * w]


# ------------------------------------------------------------------- pop / geometry

def phase_shift(a, b):
    """Sub-tile (dy, dx) that best aligns b onto a, via phase correlation. Signed, in pixels."""
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    r = fa * np.conj(fb)
    mag = np.abs(r)
    mag[mag < 1e-8] = 1e-8
    corr = np.fft.irfft2(r / mag, s=a.shape)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    dy, dx = peak
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return dy, dx


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


def silhouette_band(depth, ew, eh, edge_pct=99.3, band_px=28):
    """Depth silhouettes (top ~0.7% horizontal depth steps) upsampled to eye size, and the
    narrow disocclusion band beside them. Returns (edge, band, ref) boolean masks at eye res."""
    depth_up = resize_to(depth, ew, eh)
    gx_d = np.abs(np.diff(depth_up, axis=1, prepend=depth_up[:, :1]))
    edge = gx_d >= np.percentile(gx_d, edge_pct)
    band = hdilate(edge, band_px) & ~hdilate(edge, 2)   # beside the silhouette, not the edge itself
    ref = ~hdilate(edge, band_px + 8)                    # undisturbed regions away from silhouettes
    return edge, band, ref


def disocclusion_metrics(eye, depth):
    """From the depth silhouettes, measure the fill quality in the disocclusion band.

    disocc_frac  fraction of eye pixels in a band beside a real silhouette (how much the warp had
                 to invent). disocc_smear  horizontal-detail deficit there: 1 - |d/dx eye|(band) /
                 |d/dx eye|(clean). 0 = fill as crisp as clean regions; ->1 = smeared/stretched."""
    eh, ew = eye.shape
    _, band, ref = silhouette_band(depth, ew, eh)
    gx = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    b = float(gx[band].mean()) if band.any() else 0.0
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


def edge_accuracy(depth, src_gray, edge_pct=99.3, col_pct=97.0, maxd=24):
    """Depth-silhouette accuracy vs the true object edge (targets soft/bent/floating silhouettes).

    At depth resolution: distance from each depth silhouette to the nearest strong SOURCE color
    edge (both horizontal edges, since silhouettes are vertical). Small = the depth model's
    silhouette sits on the real object boundary; large = it floats off it. Returns p50/p95 in
    depth-px. Needs the source frame (only computed when --frames is given)."""
    dh, dw = depth.shape
    src = resize_to(src_gray, dw, dh)
    gxd = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    de = gxd >= np.percentile(gxd, edge_pct)
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
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
        out["tiles"] = int(len(dxs))

    if os.path.exists(depth_p):
        d = load_gray(depth_p)
        out["depth_spread"] = float(np.percentile(d, 95) - np.percentile(d, 5))
        out["disocc_frac"], out["disocc_smear"] = disocclusion_metrics(left, d)

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
        out["vmisalign_px"] = float(np.median(np.abs(dys)))
    if depth is not None:
        out["depth_spread"] = float(np.percentile(depth, 95) - np.percentile(depth, 5))
        out["disocc_frac"], out["disocc_smear"] = disocclusion_metrics(left, depth)
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
    srcs = sorted(glob.glob(os.path.join(frames_dir, "*.png"))) if frames_dir else []
    rows, flicks, bflicks, swims = [], [], [], []
    prev_sbs = prev_left = prev_depth = prev_src = None
    for i, p in enumerate(paths):
        depth_p = p.replace("sbs_", "depth_")
        depth = load_gray(depth_p) if os.path.exists(depth_p) else None
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
    return rows, agg


def aggregate(rows):
    keys = [k for k in rows[0] if not k.startswith("_")]
    agg = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals:
            agg[k] = float(np.mean(vals))
    agg["_n"] = len(rows)
    agg["_models"] = sorted({r.get("_model", "") for r in rows})
    return agg


# ---------------------------------------------------------------------------- main

FMT = ["pop_px_p50", "pop_px_p95", "pop_pct_p50", "vmisalign_px", "depth_spread",
       "disocc_frac", "disocc_smear"]
SEQ_FMT = ["pop_px_p50", "pop_px_p95", "vmisalign_px", "disocc_smear", "edge_acc_p50", "flicker"]
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
