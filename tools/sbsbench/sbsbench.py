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


# --------------------------------------------------------------- stretch-band proxy

def grad_mag(a, axis):
    return np.abs(np.diff(a, axis=axis, prepend=a[:, :1] if axis == 1 else a[:1, :]))


def stretch_band(eye, depth, band_px=40, edge_pct=97.0):
    """Excess horizontal smoothness in the disocclusion band next to strong depth edges.

    Disocclusion fill stretches content horizontally -> |d/dx| collapses relative to |d/dy|.
    We measure vdom = |Gy| / (|Gx| + |Gy|) inside a band around strong vertical depth edges and
    subtract the whole-frame vdom. A positive, larger value = more smeared band. Proxy only."""
    dh, dw = depth.shape
    eh, ew = eye.shape
    depth_up = np.asarray(
        Image.fromarray((depth * 255).astype(np.uint8)).resize((ew, eh), Image.BILINEAR),
        dtype=np.float32) / 255.0
    gx_d = np.abs(np.diff(depth_up, axis=1, prepend=depth_up[:, :1]))
    thr = np.percentile(gx_d, edge_pct)
    edge = gx_d >= thr
    # Horizontal dilation by band_px on both sides (disocclusion sits beside the silhouette).
    band = edge.copy()
    for s in range(1, band_px + 1):
        band[:, s:] |= edge[:, :-s]
        band[:, :-s] |= edge[:, s:]

    gx = np.abs(np.diff(eye, axis=1, prepend=eye[:, :1]))
    gy = np.abs(np.diff(eye, axis=0, prepend=eye[:1, :]))
    vdom = gy / (gx + gy + 1e-4)
    global_v = float(vdom.mean())
    band_v = float(vdom[band].mean()) if band.any() else global_v
    return band_v - global_v


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
        out["stretch_band"] = stretch_band(left, d)

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


def measure_seq_frame(path):
    """Spatial metrics for one standalone SBS frame (harness output; no depth.png)."""
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
    return out, sbs


def measure_sequence(seq_dir):
    """A harness clip: sbs_*.png in order. Per-frame spatial metrics + temporal flicker.

    flicker = frame-to-frame mean|delta| of the SBS luma (x255). On the SAME clip the real
    motion is identical across runs, so a flicker DELTA vs baseline isolates added shimmer
    (e.g. inpaint re-hallucination) -- the metric the offline sim can't produce."""
    paths = sorted(glob.glob(os.path.join(seq_dir, "sbs_*.png")))
    if not paths:
        return None
    rows = []
    flicks = []
    prev = None
    for p in paths:
        row, sbs = measure_seq_frame(p)
        if prev is not None:
            row["flicker"] = float(np.mean(np.abs(sbs - prev)) * 255.0)
            flicks.append(row["flicker"])
        rows.append(row)
        prev = sbs
    agg = aggregate(rows)
    if flicks:
        agg["flicker_p50"] = float(np.percentile(flicks, 50))
        agg["flicker_p95"] = float(np.percentile(flicks, 95))
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

FMT = ["pop_px_p50", "pop_px_p95", "pop_pct_p50", "vmisalign_px", "depth_spread", "stretch_band"]
SEQ_FMT = ["pop_px_p50", "pop_px_p95", "pop_pct_p50", "vmisalign_px", "flicker"]


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
        print(f"{'temporal':<48}flicker p50={agg['flicker_p50']:.3f} p95={agg['flicker_p95']:.3f} (x255)")


def print_diff(agg, base, fmt=FMT):
    print("\nvs baseline:")
    for k in list(fmt) + ["flicker_p50", "flicker_p95"]:
        if k in agg and k in base:
            d = agg[k] - base[k]
            pct = (d / base[k] * 100.0) if base[k] else 0.0
            arrow = "+" if d >= 0 else ""
            print(f"  {k:<16} {base[k]:>10.3f} -> {agg[k]:>10.3f}   {arrow}{d:>8.3f} ({arrow}{pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser(description="No-reference visual metrics for host SBS dumps.")
    ap.add_argument("dumps", nargs="*", help="dump_* folders, or a harness output dir with --seq")
    ap.add_argument("--glob", help="glob pattern for dump folders (quote it)")
    ap.add_argument("--seq", help="harness clip: a directory of sbs_*.png (adds temporal flicker)")
    ap.add_argument("--json", help="write the scorecard JSON here")
    ap.add_argument("--baseline", help="print deltas vs this scorecard JSON")
    args = ap.parse_args()

    if args.seq:
        res = measure_sequence(args.seq)
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
