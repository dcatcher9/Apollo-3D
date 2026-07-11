#!/usr/bin/env python3
"""Assemble the SBS A/B report directly from two run_eval.py runs (control + treatment):
control-vs-treatment bar charts (one pair per clip), the gate's verdict, and one
section per triggered issue with control/treatment crops at each issue's WORST frame.

Usage: build_report.py <control_run_dir> <treat_run_dir> <out.html>
       (run dirs = <build-dir>/sbs_eval/<label>/ containing results.json + <clip>/sbs_*.png)
"""
import base64
import glob
import html
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ctrl_dir, treat_dir, out_html = sys.argv[1], sys.argv[2], sys.argv[3]
allow_config_diff = "--allow-config-diff" in sys.argv[4:]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import sbsbench  # noqa: E402  (sbs_score, shared with run_eval)

CTRL = json.load(open(os.path.join(ctrl_dir, "results.json")))
TREAT = json.load(open(os.path.join(treat_dir, "results.json")))
THR = json.load(open(os.path.join(SCRIPT_DIR, "thresholds.json")))["metrics"]

# An A/B report may compare different code, warp, or treatment arguments, but its evidence is
# invalid if the source set, model, base config, or metric contract changed underneath it.
_SAME_CONTEXT = ["clip_set_sha1", "mode", "model", "eval_schema", "depth_step", "suite",
                 "metric_sha256"]
if not allow_config_diff:
    _SAME_CONTEXT.append("conf_sha256")
_mismatched_context = {k: (CTRL.get("meta", {}).get(k), TREAT.get("meta", {}).get(k))
                       for k in _SAME_CONTEXT
                       if CTRL.get("meta", {}).get(k) != TREAT.get("meta", {}).get(k)}
if _mismatched_context:
    raise SystemExit(f"refusing incompatible A/B report: {_mismatched_context}")

CLIPS_ROOT = CTRL.get("meta", {}).get("clips_root") or os.path.join(SCRIPT_DIR, "clips")


def source_glob(clip, frame_id):
    return glob.glob(os.path.join(CLIPS_ROOT, clip, f"frame_{frame_id:05d}.*"))


def gt_depth_path(clip, frame_id):
    paths = glob.glob(os.path.join(CLIPS_ROOT, clip, "gt_depth", f"frame_{frame_id:05d}.*"))
    return paths[0] if paths else None


def _clip_name(clip):
    # Prefer the name run_eval captured into results.json; else the repo clip's meta.json; else id.
    nm = CTRL["clips"].get(clip, {}).get("meta", {}).get("name")
    if nm:
        return nm
    mp = os.path.join(CLIPS_ROOT, clip, "meta.json")
    if os.path.exists(mp):
        try:
            return json.load(open(mp)).get("name", clip)
        except Exception:
            pass
    return clip


_NAME_CACHE = {}


def name(clip):
    """Scene display name for a clip id (from its meta.json; falls back to the id)."""
    if clip not in _NAME_CACHE:
        _NAME_CACHE[clip] = _clip_name(clip)
    return _NAME_CACHE[clip]
CLIPS = sorted(CTRL["clips"])

# metric, header, worse-is-higher, always-show, notable-threshold
COLS = [
    ("score", "score", False, True, 0),
    ("pop_spread_px", "pop_spread", False, True, 0),
    ("source_residual_p95", "warp_resid", True, True, 0),
    ("source_halo_p95", "source_halo", True, True, 0),
    ("source_stretch_pct", "source_stretch", True, True, 0),
    ("static_jitter_p95", "static_jitter", True, True, 0),
    ("flow_temporal_p95", "flow_temporal", True, True, 0),
    ("depth_gt_si_rmse", "gt_depth_rmse", True, True, 0),
    ("depth_gt_edge_f1", "gt_edge_f1", False, True, 0),
    ("positive_disparity_pct", "disp_positive", True, True, 0),
    ("negative_disparity_pct", "disp_negative", True, True, 0),
    ("source_coverage_pct", "coverage", False, True, 0),
    ("image_integrity_pct", "integrity", False, True, 0),
    ("vmisalign_px", "vmis", True, True, 0),
]

# Quality impact = the max points a metric can move the artifact score, so tables and sections
# read high-impact -> low. Score itself leads; artifacts scale by their penalty weight; stereo
# volume and context metrics remain visible but do not gain artificial score importance.
_SC = sbsbench.SCORE_CFG
_DW = _SC.get("depth", {}).get("weight", 0.0)
_PEN = _SC.get("penalties", {})
_DEPTH_METRIC = _SC.get("depth", {}).get("metric", "pop_pct_p50")


def impact(k):
    if k == "score":
        return 1e9
    if k in _PEN:
        return (1.0 - _DW) * _PEN[k]["weight"]
    if k in ("pop_spread_px", _DEPTH_METRIC):
        return _DW * 100.0
    return 0.0


COLS = sorted(COLS, key=lambda c: -impact(c[0]))
SHORT = {k: h for k, h, *_ in COLS}
def durl(im, w=None, jpg=False, q=82):
    if w and im.width > w:
        im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    b = io.BytesIO()
    if jpg:
        im.convert("RGB").save(b, "JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    im.convert("RGB").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def load_depth(p):
    return sbsbench.load_depth(p)


def frame_path(run, clip, i):
    return os.path.join(run, clip, f"sbs_{i:05d}.png")


def mid_frame(run, clip):
    n = len(glob.glob(os.path.join(run, clip, "sbs_*.png")))
    return max(0, n // 2)


def crop_at_silhouette(clip, idx):
    """Control/treatment left-eye crops at the strongest depth silhouette of frame idx (falls
    back to center if the depth is flat). Returns (ctrl_durl, treat_durl) or None."""
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    dp = os.path.join(ctrl_dir, clip, f"depth_{idx:05d}.png")
    if not (os.path.exists(cp) and os.path.exists(tp) and os.path.exists(dp)):
        return None
    depth = load_depth(dp)
    sbs_c, sbs_t = Image.open(cp), Image.open(tp)
    ew, eh = sbs_c.width // 2, sbs_c.height
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    dh, dw = depth.shape
    band = gx[int(dh * 0.15):int(dh * 0.85)]
    colscore = band.sum(0)
    lo, hi = int(dw * 0.1), int(dw * 0.9)
    cx_d = int(np.argmax(colscore[lo:hi]) + lo) if colscore[lo:hi].max() > 0.1 else dw // 2
    rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
    cy_d = int(np.argmax(rowscore)) if rowscore.max() > 0 else dh // 2
    cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, cx - cw // 2))
    y0 = max(0, min(eh - ch, cy - ch // 2))
    out = []
    for img in (sbs_c, sbs_t):
        # Crop as a modest-width JPEG; the page CSS scales it up. Keep the report light (many
        # crops embed as data URIs) so the artifact viewer loads reliably.
        out.append(durl(img.crop((x0, y0, x0 + cw, y0 + ch)), w=380, jpg=True, q=78))
    return out


def source_residual_evidence(clip, idx):
    """Metric-specific source/control/treatment crops and signed residual-delta heatmap."""
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    srcs = source_glob(clip, idx)
    if not (os.path.exists(cp) and os.path.exists(tp) and srcs):
        return None
    ctrl, treat = Image.open(cp).convert("RGB"), Image.open(tp).convert("RGB")
    ew, eh = ctrl.width // 2, ctrl.height
    src_rgb = Image.open(srcs[0]).convert("RGB").resize((ew, eh), Image.BILINEAR)
    src_gray = sbsbench.load_gray(srcs[0])
    maps = []
    for img in (ctrl, treat):
        gray = np.asarray(img.convert("L"), np.float32) / 255.0
        eyes = sbsbench.split_eyes(gray)
        maps.append([sbsbench.source_match_map(eye, src_gray)[0] for eye in eyes])
    # Show the eye where treatment changed the residual most; both eyes are always measured.
    eye_idx = max(range(2), key=lambda i: float(np.percentile(np.abs(maps[1][i] - maps[0][i]), 95)))
    delta = maps[1][eye_idx] - maps[0][eye_idx]
    score = sbsbench._box3(np.abs(delta))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, int(cx) - cw // 2))
    y0 = max(0, min(eh - ch, int(cy) - ch // 2))
    xoff = eye_idx * ew
    source_crop = src_rgb.crop((x0, y0, x0 + cw, y0 + ch))
    ctrl_crop = ctrl.crop((xoff + x0, y0, xoff + x0 + cw, y0 + ch))
    treat_crop = treat.crop((xoff + x0, y0, xoff + x0 + cw, y0 + ch))
    d = delta[y0:y0 + ch, x0:x0 + cw] * 255.0
    heat = np.zeros((*d.shape, 3), np.uint8)
    heat[..., 0] = np.clip(d * 12.0, 0, 255).astype(np.uint8)       # red = worse
    heat[..., 2] = np.clip(-d * 12.0, 0, 255).astype(np.uint8)      # blue = better
    return (durl(source_crop, w=380, jpg=True, q=82),
            durl(ctrl_crop, w=380, jpg=True, q=82),
            durl(treat_crop, w=380, jpg=True, q=82),
            durl(Image.fromarray(heat), w=380, jpg=True, q=88))


def static_jitter_evidence(clip, idx):
    """Source-static mask, per-run temporal delta, and signed treatment movement."""
    prev_idx = idx - 1
    if prev_idx < 1:
        return None
    paths = [frame_path(run, clip, i) for run in (ctrl_dir, treat_dir)
             for i in (prev_idx, idx)]
    src_now = source_glob(clip, idx)
    src_prev = source_glob(clip, prev_idx)
    if not all(os.path.exists(p) for p in paths) or not src_now or not src_prev:
        return None
    images = [Image.open(p).convert("RGB") for p in paths]
    ew, eh = images[0].width // 2, images[0].height
    stable = sbsbench.static_region_mask(
        sbsbench.load_gray(src_now[0]), sbsbench.load_gray(src_prev[0]), ew, eh)
    deltas = []
    for before, now in ((images[0], images[1]), (images[2], images[3])):
        bg = np.asarray(before.convert("L"), np.float32) / 255.0
        ng = np.asarray(now.convert("L"), np.float32) / 255.0
        be, ne = sbsbench.split_eyes(bg), sbsbench.split_eyes(ng)
        deltas.append([np.abs(ne[i] - be[i]) * stable for i in range(2)])
    eye_idx = max(range(2), key=lambda i: float(np.percentile(
        np.abs(deltas[1][i] - deltas[0][i])[stable], 95)) if stable.any() else 0.0)
    signed = (deltas[1][eye_idx] - deltas[0][eye_idx]) * 255.0
    score = sbsbench._box3(np.abs(signed))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, int(cx) - cw // 2))
    y0 = max(0, min(eh - ch, int(cy) - ch // 2))
    xoff = eye_idx * ew
    source = Image.open(src_now[0]).convert("RGB").resize((ew, eh), Image.BILINEAR)
    source_a = np.asarray(source).copy()
    source_a[~stable] = (source_a[~stable] * 0.18).astype(np.uint8)
    ctrl_heat = np.zeros((eh, ew, 3), np.uint8)
    treat_heat = np.zeros((eh, ew, 3), np.uint8)
    ctrl_heat[..., 0] = np.clip(deltas[0][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    treat_heat[..., 0] = np.clip(deltas[1][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    signed_heat = np.zeros((eh, ew, 3), np.uint8)
    signed_heat[..., 0] = np.clip(signed * 8.0, 0, 255).astype(np.uint8)
    signed_heat[..., 2] = np.clip(-signed * 8.0, 0, 255).astype(np.uint8)
    crop = (x0, y0, x0 + cw, y0 + ch)
    return tuple(durl(Image.fromarray(a).crop(crop), w=380, jpg=True, q=88) for a in
                 (source_a, ctrl_heat, treat_heat, signed_heat))


def ground_truth_depth_evidence(clip, idx):
    """Ground truth, aligned control/treatment depth, and signed error-delta map."""
    gp = gt_depth_path(clip, idx)
    cp = os.path.join(ctrl_dir, clip, f"depth_{idx:05d}.png")
    tp = os.path.join(treat_dir, clip, f"depth_{idx:05d}.png")
    if not gp or not all(os.path.exists(p) for p in (gp, cp, tp)):
        return None
    gt, control, treatment = load_depth(gp), load_depth(cp), load_depth(tp)
    kind = CTRL["clips"].get(clip, {}).get("meta", {}).get("gt_depth_kind", "disparity")
    if kind in ("metric", "depth"):
        gt, valid = sbsbench.resize_metric_depth(gt, control.shape[1], control.shape[0])
        target = np.zeros_like(gt)
        target[valid] = 1.0 / gt[valid]
    else:
        gt = sbsbench.resize_depth(gt, control.shape[1], control.shape[0])
        valid = np.isfinite(gt)
        valid &= gt >= 0.0
        target = gt
    if treatment.shape != control.shape:
        treatment = sbsbench.resize_depth(treatment, control.shape[1], control.shape[0])

    def align(pred):
        pv, tv = pred[valid], target[valid]
        t5, t95 = np.percentile(tv, (5, 95))
        if t95 - t5 < 1e-4:
            return pred + float(np.median(tv) - np.median(pv))
        design = np.column_stack((pv, np.ones_like(pv)))
        scale, shift = np.linalg.lstsq(design, tv, rcond=None)[0]
        return pred * float(scale) + float(shift)

    ca, ta = align(control), align(treatment)
    lo, hi = np.percentile(target[valid], (1, 99))
    if hi - lo < 1e-4:
        lo, hi = 0.0, 1.0
    gray = lambda a: np.clip((a - lo) / (hi - lo), 0, 1)
    signed = np.zeros_like(target)
    signed[valid] = ((np.abs(ta[valid] - target[valid]) - np.abs(ca[valid] - target[valid]))
                     * 255.0 / max(hi - lo, 1e-4))
    heat = np.zeros((*gt.shape, 3), np.uint8)
    heat[..., 0] = np.clip(signed * 5.0, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip(-signed * 5.0, 0, 255).astype(np.uint8)
    shown_gt = np.where(valid, target, lo)
    images = [Image.fromarray((gray(a) * 255).astype(np.uint8)) for a in (shown_gt, ca, ta)]
    images.append(Image.fromarray(heat))
    return tuple(durl(im, w=380, jpg=True, q=88) for im in images)


def flow_temporal_evidence(clip, idx):
    """Source-flow-compensated temporal residual for both runs and signed treatment delta."""
    if idx <= 0:
        return None
    src_now = source_glob(clip, idx)
    src_prev = source_glob(clip, idx - 1)
    paths = [frame_path(ctrl_dir, clip, idx - 1), frame_path(ctrl_dir, clip, idx),
             frame_path(treat_dir, clip, idx - 1), frame_path(treat_dir, clip, idx)]
    if not src_now or not src_prev or not all(os.path.exists(p) for p in paths):
        return None
    images = [sbsbench.load_gray(p) for p in paths]
    eyes = [sbsbench.split_eyes(a) for a in images]
    eh, ew = eyes[0][0].shape
    scale = min(1.0, 256.0 / ew)
    vw, vh = max(32, round(ew * scale)), max(24, round(eh * scale))
    now_src = sbsbench.load_gray(src_now[0])
    prev_src = sbsbench.load_gray(src_prev[0])
    gt_flow = os.path.join(CLIPS_ROOT, clip, "gt_flow", f"frame_{idx:05d}.npz")
    if os.path.exists(gt_flow):
        with np.load(gt_flow, allow_pickle=False) as flow_data:
            reference_flow = np.asarray(flow_data["flow"], dtype=np.float32)
            reference_valid = (np.asarray(flow_data["valid"], dtype=bool)
                               if "valid" in flow_data else None)
        u, v, flow_valid = sbsbench.resize_forward_flow_to_current(
            reference_flow, reference_valid, vw, vh)
    else:
        u, v, flow_valid = sbsbench.dense_source_flow(prev_src, now_src, vw, vh)
    now_small = sbsbench.resize_to(now_src, vw, vh)
    prev_small = sbsbench.resize_to(prev_src, vw, vh)
    src_warp, valid = sbsbench.warp_previous_with_flow(prev_small, u, v)
    reliable = flow_valid & valid & (np.abs(now_small - src_warp) <= 10.0 / 255.0)
    reliable &= ~sbsbench.hdilate(~reliable, 1)
    deltas = []
    for run_eyes in (eyes[:2], eyes[2:]):
        per_eye = []
        for eye_idx in range(2):
            before = sbsbench.resize_to(run_eyes[0][eye_idx], vw, vh)
            now = sbsbench.resize_to(run_eyes[1][eye_idx], vw, vh)
            warped, ok = sbsbench.warp_previous_with_flow(before, u, v)
            per_eye.append(np.abs(now - warped) * reliable * ok)
        deltas.append(per_eye)
    eye_idx = max(range(2), key=lambda i: float(np.percentile(
        np.abs(deltas[1][i] - deltas[0][i])[reliable], 95)) if reliable.any() else 0.0)
    signed = (deltas[1][eye_idx] - deltas[0][eye_idx]) * 255.0
    score = sbsbench._box3(np.abs(signed))
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    cw, ch = min(220, vw), min(170, vh)
    x0, y0 = max(0, min(vw - cw, int(cx) - cw // 2)), max(0, min(vh - ch, int(cy) - ch // 2))
    source_rgb = np.repeat((now_small[..., None] * 255.0).astype(np.uint8), 3, axis=2)
    source_rgb[~reliable] = (source_rgb[~reliable] * 0.18).astype(np.uint8)
    ctrl_heat = np.zeros((vh, vw, 3), np.uint8)
    treat_heat = np.zeros((vh, vw, 3), np.uint8)
    ctrl_heat[..., 0] = np.clip(deltas[0][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    treat_heat[..., 0] = np.clip(deltas[1][eye_idx] * 255.0 * 8.0, 0, 255).astype(np.uint8)
    signed_heat = np.zeros((vh, vw, 3), np.uint8)
    signed_heat[..., 0] = np.clip(signed * 8.0, 0, 255).astype(np.uint8)
    signed_heat[..., 2] = np.clip(-signed * 8.0, 0, 255).astype(np.uint8)
    crop = (x0, y0, x0 + cw, y0 + ch)
    return tuple(durl(Image.fromarray(a).crop(crop), w=380, jpg=True, q=88)
                 for a in (source_rgb, ctrl_heat, treat_heat, signed_heat))


def visual_evidence_images(clip, idx, metric=None):
    """Matched control/treatment crops plus an amplified RGB difference heatmap.

    The crop is selected from the shared control depth, so both modes show exactly the same
    source region.  The heatmap is deliberately labelled as amplified: it is evidence of where
    the renderers differ, while the adjacent metric supplies the direction of the change.
    """
    if metric in ("source_residual_p95", "source_halo_p95", "source_stretch_pct"):
        return source_residual_evidence(clip, idx)
    if metric == "static_jitter_p95":
        return static_jitter_evidence(clip, idx)
    if metric == "flow_temporal_p95":
        return flow_temporal_evidence(clip, idx)
    if metric in ("depth_gt_si_rmse", "depth_gt_edge_f1"):
        return ground_truth_depth_evidence(clip, idx)
    if metric == "pop_spread_px":
        cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
        dp = os.path.join(ctrl_dir, clip, f"depth_{idx:05d}.png")
        if not (os.path.exists(cp) and os.path.exists(tp) and os.path.exists(dp)):
            return None
        depth = load_depth(dp)
        ctrl, treat = Image.open(cp).convert("RGB"), Image.open(tp).convert("RGB")
        ew, eh = ctrl.width // 2, ctrl.height
        gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
        dh, dw = depth.shape
        band = gx[int(dh * 0.15):int(dh * 0.85)]
        colscore = band.sum(0)
        lo, hi = int(dw * 0.1), int(dw * 0.9)
        cx_d = int(np.argmax(colscore[lo:hi]) + lo) if colscore[lo:hi].max() > 0.1 else dw // 2
        rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
        cy_d = int(np.argmax(rowscore)) if rowscore.max() > 0 else dh // 2
        cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
        cw, ch = min(300, ew), min(300, eh)
        x0 = max(0, min(ew - cw, cx - cw // 2))
        y0 = max(0, min(eh - ch, cy - ch // 2))

        def stereo_pair(image):
            left = image.crop((x0, y0, x0 + cw, y0 + ch))
            right = image.crop((ew + x0, y0, ew + x0 + cw, y0 + ch))
            pair = Image.new("RGB", (cw * 2 + 4, ch), (18, 24, 29))
            pair.paste(left, (0, 0))
            pair.paste(right, (cw + 4, 0))
            return pair

        ctrl_pair, treat_pair = stereo_pair(ctrl), stereo_pair(treat)
        a, b = np.asarray(ctrl_pair, np.float32), np.asarray(treat_pair, np.float32)
        delta = np.mean(np.abs(b - a), axis=2)
        v = np.clip(delta * 5.0, 0, 255)
        heat = np.zeros((*v.shape, 3), np.uint8)
        heat[..., 0] = v.astype(np.uint8)
        heat[..., 1] = np.clip((v - 64) * 1.7, 0, 255).astype(np.uint8)
        return (durl(ctrl_pair, w=520, jpg=True, q=86), durl(treat_pair, w=520, jpg=True, q=86),
                durl(Image.fromarray(heat), w=520, jpg=True, q=82))
    pair = crop_at_silhouette(clip, idx)
    if not pair:
        return None
    cp, tp = frame_path(ctrl_dir, clip, idx), frame_path(treat_dir, clip, idx)
    dp = os.path.join(ctrl_dir, clip, f"depth_{idx:05d}.png")
    depth = load_depth(dp)
    ctrl, treat = Image.open(cp).convert("RGB"), Image.open(tp).convert("RGB")
    ew, eh = ctrl.width // 2, ctrl.height
    gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
    dh, dw = depth.shape
    band = gx[int(dh * 0.15):int(dh * 0.85)]
    score = band.sum(0)
    lo, hi = int(dw * 0.1), int(dw * 0.9)
    cx_d = int(np.argmax(score[lo:hi]) + lo) if score[lo:hi].max() > 0.1 else dw // 2
    rows = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
    cy_d = int(np.argmax(rows)) if rows.max() > 0 else dh // 2
    cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)
    cw, ch = min(480, ew), min(360, eh)
    x0 = max(0, min(ew - cw, cx - cw // 2))
    y0 = max(0, min(eh - ch, cy - ch // 2))
    a = np.asarray(ctrl.crop((x0, y0, x0 + cw, y0 + ch)), np.float32)
    b = np.asarray(treat.crop((x0, y0, x0 + cw, y0 + ch)), np.float32)
    delta = np.mean(np.abs(b - a), axis=2)
    # Black means unchanged. Red/yellow/white means progressively larger RGB disagreement.
    v = np.clip(delta * 5.0, 0, 255)
    heat = np.zeros((*v.shape, 3), np.uint8)
    heat[..., 0] = v.astype(np.uint8)
    heat[..., 1] = np.clip((v - 64) * 1.7, 0, 255).astype(np.uint8)
    heat[..., 2] = np.clip((v - 160) * 2.7, 0, 255).astype(np.uint8)
    return pair[0], pair[1], durl(Image.fromarray(heat), w=380, jpg=True, q=82)


def run_label(run, run_dir, default):
    """Human name for a run. The run identity must survive identical harness arguments."""
    stored = run.get("meta", {}).get("run_name")
    if stored:
        return stored
    dirname = os.path.basename(os.path.normpath(run_dir))
    if dirname:
        return dirname
    ex = run["meta"].get("extra_args") or []
    if ex:
        return " ".join(ex).replace("--", "")
    mode = run["meta"].get("mode", "")
    models = sorted({e["meta"].get("model", "") for e in run["clips"].values()})
    if mode:
        return f"{mode}" + (f" ({models[0]})" if len(models) == 1 and models[0] else "")
    return default


CTRL_MODE = CTRL["meta"].get("mode")
TREAT_MODE = TREAT["meta"].get("mode")
IS_MODE_CMP = bool(CTRL_MODE and TREAT_MODE and CTRL_MODE != TREAT_MODE)
IS_COMPARISON_ONLY = TREAT.get("verdict") == "comparison_only"
CTRL_WARPS = {e.get("meta", {}).get("warp") for e in CTRL["clips"].values()}
TREAT_WARPS = {e.get("meta", {}).get("warp") for e in TREAT["clips"].values()}
IS_WARP_CMP = CTRL_WARPS != TREAT_WARPS
IS_TRADEOFF_CMP = IS_MODE_CMP or IS_WARP_CMP
CTRL_NAME = run_label(CTRL, ctrl_dir, "control")
TREAT_NAME = run_label(TREAT, treat_dir, "treatment")
# Short tags for inline value labels and image captions (arrow is always CTRL -> TREAT).
CTRL_TAG = CTRL_MODE if IS_MODE_CMP else "control"
TREAT_TAG = TREAT_MODE if IS_MODE_CMP else "treatment"


def treatment_name():
    return TREAT_NAME


ctrl_agg = {c: CTRL["clips"][c]["aggregate"] for c in CLIPS}
treat_agg = {c: TREAT["clips"][c]["aggregate"] for c in CLIPS}

def expected_flat(run, clip):
    value = run["clips"].get(clip, {}).get("meta", {}).get("expected_flat")
    if value is not None:
        return bool(value)
    mp = os.path.join(CLIPS_ROOT, clip, "meta.json")
    try:
        return bool(json.load(open(mp)).get("expected_flat"))
    except Exception:
        return False


# Expected-flat clips remain visible as false-stereo diagnostics but cannot raise or lower the
# general-content feature verdict. They exercise a different objective from ordinary scenes.
DECISION_CLIPS = [c for c in CLIPS if not expected_flat(CTRL, c)] or CLIPS


# Re-apply eligibility to old run artifacts so a regenerated report uses today's metric contract.
for _run, _aggs in ((CTRL, ctrl_agg), (TREAT, treat_agg)):
    for _clip, _agg in _aggs.items():
        if _agg.get("disocc_frac", 0.0) < sbsbench.MIN_DISOCC_FRAC:
            for _key in ("disocc_smear", "flicker_disocc", "flicker_disocc_p50", "flicker_disocc_p95"):
                _agg.pop(_key, None)
        _agg.update(sbsbench.sbs_score(_agg, expected_flat=expected_flat(_run, _clip)))
colmax = {k: max(max(ctrl_agg[c].get(k, 0), treat_agg[c].get(k, 0)) for c in CLIPS) for k, *_ in COLS}
ACTIVE = [col for col in COLS if col[3] or colmax[col[0]] > col[4]]
CLEAN = [col for col in COLS if col not in ACTIVE and col[2]]


# Radar charts are summaries, not decision logic. Each quality axis uses a documented reference
# scale and is flipped where necessary so farther from the center always means better. The raw
# means remain printed under each chart; the real decision continues to use per-clip tolerances.
RADAR_GROUPS = [
    ("Validated primary axes", "Can vote in the feature decision", [
        {"key": "pop_spread_pct", "label": "Stereo volume", "better": "higher",
         "reference": _SC.get("depth", {}).get("target", 2.5), "unit": "%"},
        {"key": "source_residual_p95", "label": "Warp fidelity", "better": "lower",
         "reference": 15.0, "unit": " luma"},
        {"key": "source_halo_p95", "label": "Halo fidelity", "better": "lower",
         "reference": 15.0, "unit": " luma"},
        {"key": "source_stretch_pct", "label": "Stretch fidelity", "better": "lower",
         "reference": 25.0, "unit": "%"},
        {"key": "static_jitter_p95", "label": "Static stability", "better": "lower",
         "reference": 10.0, "unit": " luma"},
        {"key": "flow_temporal_p95", "label": "Motion stability", "better": "lower",
         "reference": 15.0, "unit": " luma"},
    ]),
    ("Reference validation", "Only clips with GT/reliable flow vote", [
        {"key": "depth_gt_si_rmse", "label": "GT depth accuracy", "better": "lower",
         "reference": 50.0, "unit": "%"},
        {"key": "depth_gt_edge_f1", "label": "GT boundaries", "better": "higher",
         "reference": 100.0, "unit": "%"},
        {"key": "flow_depth_p95", "label": "Flow depth stability", "better": "lower",
         "reference": 75.0, "unit": " /255"},
    ]),
]

PERF_RADAR_AXES = [
    {"key": "depth_infer", "label": "Depth speed", "better": "lower",
     "reference": 5.0, "unit": " ms"},
    {"key": "warp_infer", "label": "Warp speed", "better": "lower",
     "reference": 0.25, "unit": " ms"},
    {"key": "sbs_composite_cpu", "label": "CPU composite", "better": "lower",
     "reference": 0.05, "unit": " ms"},
]


def _mean_aggregate(aggs, key, clips=DECISION_CLIPS):
    values = [aggs[c].get(key) for c in clips if aggs[c].get(key) is not None]
    return float(np.mean(values)) if values else None


def _mean_perf(run, key):
    values = [run["clips"][c].get("perf_ms", {}).get(key) for c in CLIPS]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _radar_quality(value, axis):
    """Map a raw metric to 0..1 quality using an explicit bad-end/target reference."""
    if value is None:
        return 0.0
    ref = max(float(axis["reference"]), 1e-9)
    quality = value / ref if axis["better"] == "higher" else 1.0 - value / ref
    return max(0.0, min(1.0, quality))


def _radar_svg(title, axes, control_values, treatment_values):
    width, height, cx, cy, radius = 380, 330, 190, 150, 102
    n = len(axes)

    def point(i, scale):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        return cx + radius * scale * math.cos(angle), cy + radius * scale * math.sin(angle)

    def polygon(scales):
        return " ".join(f"{point(i, scale)[0]:.1f},{point(i, scale)[1]:.1f}"
                        for i, scale in enumerate(scales))

    rings = "".join(
        f'<polygon class="radar-ring" points="{polygon([level] * n)}" />'
        for level in (0.25, 0.5, 0.75, 1.0))
    spokes = "".join(
        f'<line class="radar-spoke" x1="{cx}" y1="{cy}" '
        f'x2="{point(i, 1)[0]:.1f}" y2="{point(i, 1)[1]:.1f}" />'
        for i in range(n))
    labels = []
    for i, axis in enumerate(axes):
        angle = -math.pi / 2 + 2 * math.pi * i / n
        x, y = cx + (radius + 22) * math.cos(angle), cy + (radius + 22) * math.sin(angle)
        anchor = "middle" if abs(math.cos(angle)) < 0.25 else "start" if math.cos(angle) > 0 else "end"
        labels.append(f'<text class="radar-label" x="{x:.1f}" y="{y + 4:.1f}" '
                      f'text-anchor="{anchor}">{html.escape(axis["label"])}</text>')
    ctrl_q = [_radar_quality(value, axis) for value, axis in zip(control_values, axes)]
    treat_q = [_radar_quality(value, axis) for value, axis in zip(treatment_values, axes)]
    title_safe = html.escape(title)
    return (f'<svg class="radar-svg" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{title_safe} radar comparison"><title>{title_safe}: outward is better</title>'
            f'{rings}{spokes}<polygon class="radar-poly radar-control" points="{polygon(ctrl_q)}" />'
            f'<polygon class="radar-poly radar-treatment" points="{polygon(treat_q)}" />'
            f'{"".join(labels)}</svg>')


def _radar_card(title, note, axes, control_values, treatment_values):
    rows = []
    for axis, control, treatment in zip(axes, control_values, treatment_values):
        c = "n/a" if control is None else f'{control:.2f}{axis["unit"]}'
        t = "n/a" if treatment is None else f'{treatment:.2f}{axis["unit"]}'
        rows.append(f'<div><span>{html.escape(axis["label"])}</span><code>{c}</code>'
                    f'<span class="radar-arrow">&rarr;</span><code>{t}</code></div>')
    return (f'<article class="radar-card"><div class="radar-head"><h3>{html.escape(title)}</h3>'
            f'<span>{html.escape(note)}</span></div>'
            f'{_radar_svg(title, axes, control_values, treatment_values)}'
            f'<div class="radar-legend"><span class="legend-control">{html.escape(CTRL_TAG)}</span>'
            f'<span class="legend-treatment">{html.escape(TREAT_TAG)}</span></div>'
            f'<div class="radar-values">{"".join(rows)}</div></article>')


def grouped_quality_section():
    cards = []
    for title, note, axes in RADAR_GROUPS:
        cards.append(_radar_card(title, note, axes,
                                [_mean_aggregate(ctrl_agg, a["key"]) for a in axes],
                                [_mean_aggregate(treat_agg, a["key"]) for a in axes]))
    cards.append(_radar_card("Runtime", "Performance context; not a quality vote", PERF_RADAR_AXES,
                            [_mean_perf(CTRL, a["key"]) for a in PERF_RADAR_AXES],
                            [_mean_perf(TREAT, a["key"]) for a in PERF_RADAR_AXES]))

    hard_defs = (
        ("vmisalign_px", "Vertical alignment", " px"),
        ("positive_disparity_pct", "Positive disparity tail", "%"),
        ("negative_disparity_pct", "Negative disparity tail", "%"),
        ("source_coverage_pct", "Source coverage", "%"),
        ("image_integrity_pct", "Image integrity", "%"),
    )
    checks = []
    for key, label, unit in hard_defs:
        spec = THR[key]
        control = _mean_aggregate(ctrl_agg, key, CLIPS)
        treatment = _mean_aggregate(treat_agg, key, CLIPS)
        bound = (f'≥ {spec["hard_min"]:.1f}{unit}' if "hard_min" in spec
                 else f'≤ {spec["hard_max"]:.1f}{unit}')
        def value(v):
            if v is None:
                return "n/a", "hard-missing"
            failed = (("hard_min" in spec and v < spec["hard_min"])
                      or ("hard_max" in spec and v > spec["hard_max"]))
            return f"{v:.2f}{unit}", "hard-fail" if failed else "hard-pass"
        cv, cc = value(control)
        tv, tc = value(treatment)
        checks.append(f'<div class="hard-check"><span>{label}</span><code class="{cc}">{cv}</code>'
                      f'<span class="radar-arrow">&rarr;</span><code class="{tc}">{tv}</code>'
                      f'<small>{bound}</small></div>')
    hard_card = (f'<article class="hard-card"><div><h3>Hard constraints: comfort and integrity</h3>'
                 f'<p>Every row must pass independently; quality improvements cannot trade '
                 f'against a failed limit.</p></div><div class="hard-checks">'
                 f'{"".join(checks)}</div></article>')
    return (f'<section><h2>Metrics by group</h2><p class="sub">Radar axes are normalized quality: '
            f'<b>farther outward is always better</b>. Means use the non-flat decision clips; '
            f'runtime uses all clips. The reference scale is the stereo target or the metric\'s '
            f'documented penalty/engineering scale, never the best value in this A/B pair. Raw '
            f'means are printed below every chart. These summaries do not replace the per-clip gate.</p>'
            f'<div class="radar-grid">{"".join(cards)}</div>{hard_card}</section>')

def scorecard_charts():
    """Grouped horizontal bars retain every table value while making A/B movement scannable."""
    charts = []
    for metric, label, worse, _, _ in ACTIVE:
        values = [(c, ctrl_agg[c].get(metric), treat_agg[c].get(metric)) for c in CLIPS]
        numeric = [abs(v) for _, a, b in values for v in (a, b) if v is not None]
        scale = max(numeric, default=1.0) or 1.0
        rows = []
        for c, a, b in values:
            if a is None or b is None:
                rows.append(f'<div class="bar-row"><div class="bar-scene" title="{c}">{name(c)}</div>'
                            f'<div class="bar-pair"><span class="bar-na">not applicable</span></div>'
                            f'<span class="bar-delta bar-flat">n/a</span></div>')
                continue
            aw = max(0.8, abs(a) / scale * 100.0) if a else 0.0
            bw = max(0.8, abs(b) / scale * 100.0) if b else 0.0
            delta = b - a
            floor = THR.get(metric, {}).get("abs_floor", 0.0) / 2.0
            flat = abs(delta) < max(floor, abs(a) * 0.05)
            better = (delta < 0) if worse else (delta > 0)
            move_cls = "bar-flat" if flat else "bar-good" if better else "bar-bad"
            pct = delta / a * 100.0 if a else (100.0 if b else 0.0)
            delta_text = "within noise" if flat else f'{"better" if better else "worse"} {abs(pct):.0f}%'
            rows.append(
                f'<div class="bar-row"><div class="bar-scene" title="{c}">{name(c)}</div>'
                f'<div class="bar-pair"><div class="bar-line"><span class="bar-tag">{CTRL_TAG}</span>'
                f'<span class="bar-track"><i class="bar-fill bar-control" style="width:{aw:.1f}%"></i></span>'
                f'<b>{a:.2f}</b></div><div class="bar-line"><span class="bar-tag">{TREAT_TAG}</span>'
                f'<span class="bar-track"><i class="bar-fill {move_cls}" style="width:{bw:.1f}%"></i></span>'
                f'<b>{b:.2f}</b></div></div><span class="bar-delta {move_cls}">{delta_text}</span></div>')
        direction = "lower is better" if worse else "higher is better"
        charts.append(f'<article class="metric-chart"><div class="chart-head">'
                      f'<h3>{mtip(metric, label)}</h3><span>{direction}</span></div>{"".join(rows)}</article>')
    return '<div class="chart-grid">' + "".join(charts) + '</div>'


# metric -> (short header, what it measures, direction). Only the ones that appear render.
METRIC_DEFS = [
    ("score", "score", "Overall 0-100 artifact cleanliness after weighted penalties. Stereo volume is reported and gated separately, so it cannot cancel artifact regressions.", "higher = better"),
    ("pop_spread_px", "pop_spread", "Near-to-far range of horizontal stereo disparity. This is the validated stereo-volume gate; the radar displays its resolution-independent percentage form.", "higher = more stereo volume"),
    ("positive_disparity_pct", "disp_positive", "Weighted p99 of the positive signed L/R disparity tail as a percentage of eye width. Kept sign-explicit because host output lacks headset angular calibration.", "must stay below comfort limit"),
    ("negative_disparity_pct", "disp_negative", "Magnitude of the weighted p1 negative signed L/R disparity tail as a percentage of eye width.", "must stay below comfort limit"),
    ("source_coverage_pct", "coverage", "Worst-eye interior pixels whose output patch is explained by some same-scanline source patch within the allowed stereo displacement.", "must remain above integrity limit"),
    ("image_integrity_pct", "integrity", "Worst-eye retention of source texture after horizontal source alignment. Detects missing, black, or collapsed image regions.", "must remain above integrity limit"),
    ("source_residual_p95", "warp_resid", "Worst-eye patch difference from the source after allowing a small horizontal stereo displacement. Detects monocular warp corruption without penalizing intended parallax.", "lower = more source-faithful"),
    ("source_halo_p95", "source_halo", "Excess thin-ridge brightness at depth silhouettes after subtracting the horizontally aligned source ridge. Genuine source outlines are free.", "lower = less warp-created halo"),
    ("source_stretch_pct", "source_stretch", "Source-textured silhouette-near pixels whose horizontal detail collapses below 35% of the aligned source detail.", "lower = less warp stretch"),
    ("static_jitter_p95", "static_jitter", "Worst-eye temporal change over regions whose source neighborhood stayed static after allowing for horizontal disparity. Camera/object motion is excluded.", "lower = steadier static content"),
    ("flow_temporal_p95", "flow_temporal", "Worst-eye temporal residual after warping the previous output with exact dataset flow when available, otherwise classical source flow, and rejecting photometrically unreliable samples.", "lower = steadier moving content"),
    ("depth_gt_si_rmse", "gt_depth_rmse", "Prediction error against committed ground-truth inverse depth after monocular scale/shift alignment; constant-depth GT uses shift-only alignment.", "lower = more accurate depth"),
    ("depth_gt_edge_f1", "gt_edge_f1", "Depth-boundary F1 against committed ground truth with one-pixel tolerance.", "higher = more accurate boundaries"),
    ("flow_depth_p95", "flow_depth", "Pre-warp depth change after source optical-flow compensation, on photometrically reliable support.", "lower = steadier depth"),
    ("depth_spread", "dspread", "p95−p5 of the normalized depth = pop available at the source.", "higher = more depth to work with"),
    ("edge_acc_p50", "edge_acc", "Distance (depth-px) from each depth silhouette to the nearest true SOURCE color edge.", "lower = silhouette sits on the real edge"),
    ("swim_p50", "swim", "Frame-to-frame depth change where the SOURCE is static — depth instability, separated from real motion.", "lower = steadier depth"),
    ("disocc_smear", "smear", "Horizontal-detail deficit in the narrow band beside silhouettes; on flat content also fingerprints hallucinated depth edges.", "lower = crisper fill"),
    ("flicker_disocc_p50", "flick_dis", "Flicker restricted to the disocclusion bands — inpaint/stretch re-hallucination shimmer.", "lower = less boiling along edges"),
    ("vmisalign_px", "vmis", "Median vertical L↔R offset — parallax must be horizontal-only, so this is a geometry correctness check.", "must be ≈ 0"),
]
_ROLE_ORDER = {"hard": 0, "primary": 1, "diagnostic": 2, "reported": 3}


def metric_group(key):
    spec = THR.get(key, {})
    role = spec.get("role", "reported")
    axis = spec.get("axis", "summary" if key == "score" else "stereo")
    return role, axis


METRIC_DEFS = sorted(METRIC_DEFS,
                     key=lambda m: (_ROLE_ORDER[metric_group(m[0])[0]], metric_group(m[0])[1],
                                    -impact(m[0])))


DEF_BY_KEY = {k: (what, d) for k, h, what, d in METRIC_DEFS}


def tip_text(metric):
    d = DEF_BY_KEY.get(metric)
    return f"{d[0]} ({d[1]})".replace('"', "'") if d else ""


def mtip(metric, label):
    """Metric label wrapped with a native-title tooltip (reliable inside the scroll container)."""
    t = tip_text(metric)
    return f'<span class="mtip" title="{t}">{label}</span>' if t else label


def metrics_section():
    present = ({k for aggs in (ctrl_agg, treat_agg) for agg in aggs.values() for k in agg}
               | {i["metric"] for i in CTRL["issues"]})
    rows = "".join(
        f'<tr><td class="mgroup"><span>{metric_group(k)[0]}</span><small>{metric_group(k)[1]}</small></td>'
        f'<td class="mname">{h}</td><td class="mwhat">{what}</td><td class="mdir">{d}</td></tr>'
        for k, h, what, d in METRIC_DEFS if k in present)
    return (f'<details class="fold metric-defs"><summary>Metric definitions and decision roles</summary>'
            f'<div class="fold-body"><p class="sub">Hard '
            f'constraints can reject; primary metrics can vote; diagnostics provide supporting '
            f'evidence; reported values are context only. All are computed on the real '
            f'SBS frames the headset would receive (no CPU replica). Absolute values are '
            f'resolution-dependent, so compare within a run, not across clip sets.</p>'
            f'<div class="tablewrap"><table class="mtab"><thead><tr><th>group / axis</th><th>metric</th>'
            f'<th>what it measures</th><th>direction</th></tr></thead><tbody>{rows}</tbody></table>'
            f'</div></div></details>')


def conclusion_section():
    """Auto-derived verdict using per-clip metric gates; means summarize but never decide."""
    sc_a = np.mean([ctrl_agg[c].get("score", 0) for c in DECISION_CLIPS])
    sc_b = np.mean([treat_agg[c].get("score", 0) for c in DECISION_CLIPS])
    score_line = (f'<li class="c-score">Artifact score (0-100, diagnostic mean): '
                  f'{CTRL_TAG} <b>{sc_a:.1f}</b> '
                  f'&rarr; {TREAT_TAG} <b>{sc_b:.1f}</b> ({sc_b - sc_a:+.1f})</li>')
    wins, costs = [], []
    for k, h, worse, _, _ in COLS:
        if k == "score":  # the headline, not a component metric
            continue
        a = _mean_aggregate(ctrl_agg, k)
        b = _mean_aggregate(treat_agg, k)
        if a is None or b is None:
            continue
        if a < 1e-6 and b < 1e-6:
            continue
        pct = (b - a) / a * 100 if a else 100.0
        # Significant = both a relative move AND an absolute one (half the gate's abs_floor),
        # so sub-pixel noise on near-zero metrics doesn't read as a headline.
        floor = THR.get(k, {}).get("abs_floor", 0.0) / 2.0
        if abs(pct) < 5 or abs(b - a) < floor:
            continue
        # In a mode comparison neither direction is "better/worse" globally (it's a tradeoff);
        # split by which run each metric favors instead.
        favors_treat = (pct < 0) if worse else (pct > 0)
        txt = f"{mtip(k, '<b>' + h + '</b>')} {CTRL_TAG} {a:.2f} → {TREAT_TAG} {b:.2f} ({pct:+.0f}%)"
        (wins if favors_treat else costs).append(txt)
    li = score_line
    decision = sbsbench.evaluate_ab_decision(
        ctrl_agg, treat_agg, DECISION_CLIPS, THR, hard_clip_ids=CLIPS)
    if IS_TRADEOFF_CMP:
        if wins:
            li += f'<li class="c-win">{TREAT_NAME} is better on: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">{CTRL_NAME} is better on: {" · ".join(costs)}</li>'
        verdict = (f"<b>Geometry tradeoff:</b> compare the per-metric and per-clip evidence; a "
                   f"single scalar does not select between different warp objectives.")
    else:
        if wins:
            li += f'<li class="c-win">Mean diagnostics favor treatment: {" · ".join(wins)}</li>'
        if costs:
            li += f'<li class="c-cost">Mean diagnostics favor control: {" · ".join(costs)}</li>'
        axis_parts = []
        for axis, movement in sorted(decision["axes"].items()):
            axis_parts.append(f'<b>{axis}</b>: {len(movement["improved"])} win(s), '
                              f'{len(movement["regressed"])} cost(s)')
        if axis_parts:
            li += f'<li class="c-score">Primary axes: {" · ".join(axis_parts)}</li>'
        state = decision["verdict"]
        if state == "reject_hard":
            verdict = (f'<b>Reject treatment:</b> {len(decision["hard_failures"])} hard comfort/'
                       f'integrity constraint(s) fail.')
        elif state == "reject_primary":
            verdict = (f'<b>Reject treatment:</b> {decision["regressed"]} primary-axis cost(s) '
                       f'with no compensating primary-axis win.')
        elif state == "tradeoff":
            verdict = (f'<b>Primary-quality tradeoff:</b> coequal axes move in different or mixed '
                       f'directions. Per-clip event counts are evidence, not weights. Do not '
                       f'resolve this with the scalar score; use visual/headset evidence.')
        elif state == "candidate":
            verdict = (f'<b>Candidate improvement:</b> {decision["improved"]} primary-axis win(s), '
                       f'no primary-axis costs and no hard failure.')
        else:
            verdict = ("<b>No validated decision:</b> hard constraints pass, but all validated "
                       "primary metrics remain within noise. Diagnostic proxies cannot vote.")
    head = (f"{CTRL_NAME} → {TREAT_NAME}" if IS_TRADEOFF_CMP else f"Treatment: <b>{treatment_name()}</b>")
    return (f'<section><h2>Conclusion</h2>'
            f'<p class="sub" style="margin-bottom:12px">{head} — decision over '
            f'{len(DECISION_CLIPS)} non-flat clip(s); expected-flat diagnostics remain below.</p>'
            f'<ul class="concl">{li}<li>{verdict}</li></ul>{gate_strip()}</section>')


def gate_strip():
    hard = TREAT.get("hard_failures", [])
    if hard:
        items = "".join(
            f'<li><code>{name(r["clip"])}.{r["metric"]}</code> = {r["value"]}</li>' for r in hard)
        return (f'<div class="gate gate-fail"><b>Gate: {len(hard)} HARD COMFORT/INTEGRITY '
                f'FAILURE(S)</b><ul>{items}</ul></div>')
    if IS_COMPARISON_ONLY:
        return ('<div class="gate gate-info"><b>Gate: COMPARISON ONLY</b> — committed baselines '
                'were not consulted; conclusions come from this matched control/treatment pair.</div>')
    regs = TREAT.get("regressions", [])
    noun = "difference(s) vs " + CTRL_MODE + " baseline" if IS_MODE_CMP else "regression(s)"
    if not regs:
        return ('<div class="gate gate-pass"><b>Gate: PASS</b> — no '
                + noun + ' past threshold (run_eval exit 0).</div>')
    arrow = "→"
    items = "".join(f'<li><code>{name(r["clip"])}.{r["metric"]}</code> {r["baseline"]} {arrow} {r["value"]}'
                    + (f' <span class="wf">worst frame {r["frame"]}</span>' if "frame" in r else "")
                    + "</li>" for r in regs)
    cls = "gate-fail" if not IS_MODE_CMP else "gate-info"
    label = (f"{len(regs)} {noun}" if IS_MODE_CMP else f"{len(regs)} REGRESSION(S) — run_eval exit 1")
    return f'<div class="gate {cls}"><b>Gate: {label}</b><ul>{items}</ul></div>'


def _evidence_card(item, kind, axis=None):
    """Render one metric-specific matched visual card."""
    _, delta, metric, c, a, b = item
    better = THR.get(metric, {}).get("better", "lower")
    treatment_worse = (delta > 0) if better == "lower" else (delta < 0)
    source = TREAT if treatment_worse else CTRL
    wf = source["clips"][c].get("worst_frame", {}).get(metric, {})
    frame = wf.get("frame", mid_frame(ctrl_dir, c))
    imgs = visual_evidence_images(c, frame, metric)
    if not imgs:
        return ""
    pct = delta / a * 100 if a else 100.0
    cls = ("evidence-cost" if kind == "regression" else "evidence-win" if kind == "improvement"
           else "evidence-noise")
    badge = kind.replace("_", " ")
    is_gt = metric in ("depth_gt_si_rmse", "depth_gt_edge_f1")
    source_label = ("source · bright = evaluated static region" if metric == "static_jitter_p95"
                    else "source · bright = reliable optical flow" if metric == "flow_temporal_p95"
                    else "ground-truth depth" if is_gt else "source")
    ctrl_label = (f"{CTRL_TAG} · temporal change" if metric == "static_jitter_p95" else
                  f"{CTRL_TAG} · flow residual" if metric == "flow_temporal_p95" else
                  f"{CTRL_TAG} · aligned depth" if is_gt else
                  f"{CTRL_TAG} · left | right" if metric == "pop_spread_px" else CTRL_TAG)
    treat_label = (f"{TREAT_TAG} · temporal change" if metric == "static_jitter_p95" else
                   f"{TREAT_TAG} · flow residual" if metric == "flow_temporal_p95" else
                   f"{TREAT_TAG} · aligned depth" if is_gt else
                   f"{TREAT_TAG} · left | right" if metric == "pop_spread_px" else TREAT_TAG)
    panels = (f'<div class="triplet"><figure><span class="tag">{source_label}</span><img src="{imgs[0]}"></figure>'
              f'<figure><span class="tag">{ctrl_label}</span><img src="{imgs[1]}"></figure>'
              f'<figure><span class="tag t-treat">{treat_label}</span><img src="{imgs[2]}"></figure>'
              f'<figure><span class="tag t-diff">delta: red worse / blue better</span>'
              f'<img src="{imgs[3]}"></figure></div>' if len(imgs) == 4 else
              f'<div class="triplet"><figure><span class="tag">{ctrl_label}</span>'
              f'<img src="{imgs[0]}"></figure><figure><span class="tag t-treat">{treat_label}</span>'
              f'<img src="{imgs[1]}"></figure><figure><span class="tag t-diff">abs diff &times;5</span>'
              f'<img src="{imgs[2]}"></figure></div>')
    axis_label = f'<span class="axis-label">{axis}</span>' if axis else ""
    return (f'<article class="evidence-card {cls}"><div class="ic-head">{axis_label}'
            f'<span class="clipname">{name(c)}</span><span class="pill">{badge}</span>'
            f'<span class="metricval">{mtip(metric, SHORT.get(metric, metric))}: '
            f'<b>{a:.2f}</b> &rarr; {b:.2f} ({pct:+.0f}%) &middot; frame {frame}</span></div>'
            f'{panels}</article>')


def _strongest_change(metric, clips=DECISION_CLIPS):
    floor = THR.get(metric, {}).get("abs_floor", 0.0)
    candidates = []
    for c in clips:
        a, b = ctrl_agg[c].get(metric), treat_agg[c].get(metric)
        if a is None or b is None:
            continue
        delta = b - a
        denom = max(floor, abs(a) * 0.05, 1e-6)
        candidates.append((abs(delta) / denom, delta, metric, c, a, b))
    return max(candidates, default=None)


def _change_kind(item):
    _, delta, metric, _, a, _ = item
    spec = THR[metric]
    significant = abs(delta) > max(spec.get("abs_floor", 0.0), abs(a) * spec.get("rel_tol", 0.0))
    if not significant:
        return "within_noise"
    improved = delta > 0 if spec.get("better") == "higher" else delta < 0
    return "improvement" if improved else "regression"


def visual_evidence_section():
    """Show one representative example for every validated primary quality axis."""
    axes = (("Stereo volume", ("pop_spread_px",)),
            ("Warp fidelity", ("source_residual_p95", "source_halo_p95", "source_stretch_pct")),
            ("Temporal stability", ("static_jitter_p95", "flow_temporal_p95")),
            ("Ground-truth depth", ("depth_gt_si_rmse", "depth_gt_edge_f1")))
    cards = []
    for axis, metrics in axes:
        item = max((item for metric in metrics if (item := _strongest_change(metric))),
                   default=None)
        if item:
            cards.append(_evidence_card(item, _change_kind(item), axis))
    cards = "".join(cards)
    if not cards:
        return ""
    return (f'<section><h2>Primary-axis visual evidence</h2>'
            f'<p class="sub">One strongest matched example for each decision axis. Warp and '
            f'temporal metrics use source-relative heatmaps, stereo shows both eyes, and reference '
            f'depth shows aligned prediction against ground truth. A within-noise badge means the '
            f'example is illustrative, not a decision event.</p>{cards}</section>')


def diagnostic_evidence_section():
    """Only surface unusually large diagnostic moves after the primary evidence."""
    metrics = ("stretch_area", "rim_over_p95", "edge_acc_p50", "disocc_smear",
               "flicker_disocc_p50", "swim_p50")
    candidates = []
    for metric in metrics:
        item = _strongest_change(metric)
        if not item:
            continue
        _, delta, _, _, a, _ = item
        floor = THR[metric].get("abs_floor", 0.0)
        if abs(delta) >= max(floor * 2.0, abs(a) * 0.5):
            candidates.append(item)
    cards = "".join(_evidence_card(item, _change_kind(item), "Diagnostic exception")
                    for item in sorted(candidates, reverse=True)[:3])
    if not cards:
        return ""
    return (f'<section><h2>Large diagnostic changes</h2>'
            f'<p class="sub">Shown only when a supporting metric changes by at least 50% and twice '
            f'its absolute noise floor. These examples explain a large secondary movement but do '
            f'not vote against the primary axes.</p>{cards}</section>')


def clean_footer():
    if not CLEAN:
        return ""
    items = ", ".join(f"{h} {colmax[k]:.2f}" for k, h, *_ in CLEAN)
    return (f'<p class="foot"><b>Clean this run (max ≈ 0, collapsed):</b> {items}. Still measured '
            f'every run — any one auto-appears as a column when it crosses threshold.</p>')


meta = CTRL["meta"]

HTML = """<style>
:root{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;
  --accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;
  --mono:ui-monospace,"SF Mono","Cascadia Mono",Consolas,monospace;--sans:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
@media (prefers-color-scheme:dark){:root{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;
  --line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}}
:root[data-theme="light"]{--bg:#f5f6f7;--panel:#fff;--ink:#12181d;--muted:#5c6a74;--line:#dbe1e6;--accent:#0e8f9c;--accent-soft:#d7eef0;--good:#1f9d63;--warn:#c98a1a;--crit:#c4483a;}
:root[data-theme="dark"]{--bg:#0c1013;--panel:#141a1f;--ink:#e8edf0;--muted:#8b9aa4;--line:#252e35;--accent:#38c0cd;--accent-soft:#123037;--good:#3fca86;--warn:#e0a94a;--crit:#e56a5c;}
*{box-sizing:border-box}
.wrap{max-width:1060px;margin:0 auto;padding:56px 24px 96px;color:var(--ink);font-family:var(--sans);line-height:1.6;background:var(--bg);-webkit-font-smoothing:antialiased}
h1,h2{text-wrap:balance;line-height:1.15;margin:0}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin-bottom:14px}
h1{font-size:36px;font-weight:680;letter-spacing:-.02em}
.lede{color:var(--muted);font-size:16.5px;max-width:68ch;margin-top:14px}
.meta{font-family:var(--mono);font-size:12.5px;color:var(--muted);margin-top:20px;display:flex;gap:18px;flex-wrap:wrap;border-top:1px solid var(--line);padding-top:16px}
section{margin-top:52px}
h2{font-size:15px;font-family:var(--mono);letter-spacing:.03em;text-transform:uppercase;color:var(--ink);padding-bottom:12px;border-bottom:1px solid var(--line);margin-bottom:8px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.sub{color:var(--muted);font-size:14px;margin:0 0 20px;max-width:72ch}
.foot{margin-top:14px;color:var(--muted);font-size:13px}.foot b{color:var(--ink)}
.fold{margin-top:26px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.fold>summary{cursor:pointer;list-style:none;padding:14px 17px;font-family:var(--mono);font-size:12px;font-weight:650;color:var(--ink);display:flex;align-items:center;gap:10px}.fold>summary::-webkit-details-marker{display:none}.fold>summary:before{content:"+";color:var(--accent);font-size:16px;line-height:1}.fold[open]>summary:before{content:"−"}.fold[open]>summary{border-bottom:1px solid var(--line)}
.fold-body{padding:18px}.fold-body>.sub:last-child{margin-bottom:0}.metric-defs{margin-top:28px}
.gate{border-radius:10px;padding:14px 18px;font-size:14px;margin-top:26px;border:1px solid var(--line)}
.gate-pass{background:color-mix(in srgb,var(--good) 9%,transparent);border-color:color-mix(in srgb,var(--good) 40%,var(--line))}
.gate-fail{background:color-mix(in srgb,var(--crit) 8%,transparent);border-color:color-mix(in srgb,var(--crit) 40%,var(--line))}
.gate-info{background:color-mix(in srgb,var(--accent) 8%,transparent);border-color:color-mix(in srgb,var(--accent) 40%,var(--line))}
.gate ul{margin:8px 0 0;padding-left:20px}.gate li{margin:2px 0}
.gate .wf{font-family:var(--mono);font-size:11.5px;color:var(--muted)}
.concl{margin:0 0 18px;padding-left:20px;font-size:14.5px}
.concl li{margin:7px 0;max-width:78ch}
.concl b{color:var(--ink)}
.c-win{color:var(--good)}.c-win b{color:var(--good)}
.c-cost{color:var(--crit)}.c-cost b{color:var(--crit)}
.c-score{font-size:15.5px;color:var(--ink)}.c-score b{color:var(--accent);font-size:16px}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:right;padding:11px 13px;border-bottom:1px solid var(--line);white-space:nowrap;vertical-align:middle}
th:first-child,td:first-child{text-align:left;min-width:225px}
thead th{font-family:var(--mono);font-size:11px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted);font-weight:600;background:var(--panel)}
thead th[title]:not([title=""]),.mtip{cursor:help;text-decoration:underline dotted;text-underline-offset:3px;text-decoration-color:color-mix(in srgb,var(--muted) 60%,transparent)}
tbody tr:last-child td{border-bottom:none}
td{font-family:var(--mono);font-variant-numeric:tabular-nums}
.mtab td,.mtab th{text-align:left;white-space:normal}
.mtab .mname{font-family:var(--mono);font-size:12.5px;color:var(--accent);font-weight:600;white-space:nowrap;vertical-align:top}
.mtab .mwhat{font-family:var(--sans);font-size:13.5px;color:var(--ink);max-width:60ch}
.mtab .mdir{font-family:var(--mono);font-size:11.5px;color:var(--muted);white-space:nowrap;vertical-align:top}
.mtab .mgroup{font-family:var(--mono);white-space:nowrap;vertical-align:top}.mgroup span{display:block;font-size:11.5px;color:var(--ink);font-weight:650;text-transform:uppercase}.mgroup small{display:block;font-size:10.5px;color:var(--muted)}
.radar-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}
.radar-card{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:15px;min-width:0}
.radar-head h3,.hard-card h3{font-family:var(--mono);font-size:13px;color:var(--ink);margin:0}.radar-head>span{display:block;font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:3px}
.radar-svg{width:100%;height:auto;display:block;margin:2px auto -12px;overflow:visible}
.radar-ring{fill:none;stroke:var(--line);stroke-width:1;stroke-dasharray:3 4}.radar-spoke{stroke:var(--line);stroke-width:1}.radar-label{font-family:var(--mono);font-size:10px;fill:var(--muted)}
.radar-poly{stroke-width:2;stroke-linejoin:round}.radar-control{fill:color-mix(in srgb,var(--accent) 14%,transparent);stroke:var(--accent)}.radar-treatment{fill:color-mix(in srgb,var(--warn) 14%,transparent);stroke:var(--warn)}
.radar-legend{display:flex;justify-content:center;gap:20px;font-family:var(--mono);font-size:10.5px;margin-bottom:10px}.radar-legend span:before{content:"";display:inline-block;width:14px;height:3px;border-radius:3px;margin-right:6px;vertical-align:middle}.legend-control:before{background:var(--accent)}.legend-treatment:before{background:var(--warn)}
.radar-values{border-top:1px solid var(--line);padding-top:8px}.radar-values>div{display:grid;grid-template-columns:minmax(0,1fr) auto 14px auto;align-items:center;gap:4px;font-family:var(--mono);font-size:9.5px;padding:2px 0}.radar-values>div>span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}.radar-values code{font-size:9px;padding:0 3px}.radar-arrow{color:var(--muted);text-align:center}
.hard-card{display:grid;grid-template-columns:minmax(220px,.65fr) minmax(420px,1.35fr);gap:24px;align-items:center;border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:16px;margin-top:16px}.hard-card p{font-size:12px;color:var(--muted);margin:5px 0 0}.hard-checks{display:grid;gap:5px}.hard-check{display:grid;grid-template-columns:minmax(145px,1fr) 74px 14px 74px 66px;gap:5px;align-items:center;font-family:var(--mono);font-size:10px}.hard-check code{font-size:9.5px;text-align:right;padding:1px 4px}.hard-check small{text-align:right;color:var(--muted)}.hard-pass{color:var(--good)}.hard-fail{color:var(--crit)}.hard-missing{color:var(--muted)}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.metric-chart{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:14px 14px 10px;min-width:0}
.chart-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:9px}
.chart-head h3{font-family:var(--mono);font-size:13px;color:var(--ink);margin:0}.chart-head>span{font-family:var(--mono);font-size:10px;color:var(--muted)}
.bar-row{display:grid;grid-template-columns:112px minmax(130px,1fr) 72px;align-items:center;gap:8px;padding:6px 0;border-top:1px solid color-mix(in srgb,var(--line) 65%,transparent)}
.bar-scene{font-family:var(--mono);font-size:10.5px;color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-pair{display:grid;gap:3px}.bar-line{display:grid;grid-template-columns:34px minmax(50px,1fr) 42px;align-items:center;gap:5px}
.bar-tag{font-family:var(--mono);font-size:9px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar-line b{font-family:var(--mono);font-size:9.5px;font-weight:500;text-align:right;font-variant-numeric:tabular-nums}
.bar-track{height:5px;background:color-mix(in srgb,var(--muted) 10%,transparent);border-radius:5px;overflow:hidden}
.bar-fill{display:block;height:100%;min-width:0;border-radius:5px}.bar-control{background:var(--accent)}
.bar-good{background:var(--good)}.bar-bad{background:var(--crit)}.bar-flat{background:var(--muted)}
.bar-delta{font-family:var(--mono);font-size:9.5px;text-align:right;background:none}.bar-delta.bar-good{color:var(--good)}.bar-delta.bar-bad{color:var(--crit)}.bar-delta.bar-flat{color:var(--muted)}
.bar-na{font-family:var(--mono);font-size:9.5px;color:var(--muted);align-self:center}
.clipname{font-family:var(--mono);font-size:13px;font-weight:600;color:var(--ink)}
.pill{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:20px;font-weight:600;white-space:nowrap}
.p-warn{color:var(--warn);background:color-mix(in srgb,var(--warn) 15%,transparent)}
.p-crit{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.p-info{color:var(--accent);background:var(--accent-soft)}
.issue-clip{margin-top:20px}
.ic-head{display:flex;align-items:baseline;gap:14px;margin-bottom:10px;flex-wrap:wrap}
.metricval{font-family:var(--mono);font-size:13px;color:var(--muted)}.metricval b{color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.pair.single{grid-template-columns:1fr;max-width:540px}
.pair figure{margin:0;position:relative}
.pair img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}
.evidence-card{margin-top:22px;padding:14px;border:1px solid var(--line);border-radius:11px;background:var(--panel)}
.evidence-card .pill{color:var(--good);background:color-mix(in srgb,var(--good) 15%,transparent)}
.evidence-card.evidence-cost .pill{color:var(--crit);background:color-mix(in srgb,var(--crit) 15%,transparent)}
.evidence-card.evidence-noise .pill{color:var(--muted);background:color-mix(in srgb,var(--muted) 14%,transparent)}
.axis-label{font-family:var(--mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);padding-right:4px}
.triplet{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.triplet figure{margin:0;position:relative}.triplet img{width:100%;border-radius:8px;border:1px solid var(--line);display:block}
.tag.t-diff{color:var(--accent)}
.tag{position:absolute;top:8px;left:8px;font-family:var(--mono);font-size:11px;font-weight:600;padding:2px 8px;border-radius:5px;background:color-mix(in srgb,var(--bg) 82%,transparent);border:1px solid var(--line);color:var(--ink)}
.tag.t-treat{color:var(--warn)}
pre{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px;overflow-x:auto;color:var(--ink);line-height:1.7}
code{font-family:var(--mono);font-size:12px;background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:5px}
@media (max-width:900px){.radar-grid{grid-template-columns:1fr 1fr}.radar-card:last-child{grid-column:1/-1;max-width:480px}.chart-grid{grid-template-columns:1fr}}
@media (max-width:640px){.radar-grid{grid-template-columns:1fr}.radar-card:last-child{grid-column:auto;max-width:none}.hard-card{grid-template-columns:1fr}.pair,.triplet{grid-template-columns:1fr}.bar-row{grid-template-columns:88px minmax(110px,1fr) 62px}h1{font-size:28px}}
</style>

<div class="wrap">
  <div class="eyebrow">Apollo SBS 3D &middot; run_eval A/B report</div>
  <h1>__H1__</h1>
  <p class="lede">Generated from two <code>run_eval.py</code> runs over the committed clip set —
  the real pipeline and gated metrics. __LEDE__</p>
  <div class="meta"><span>__DATE__</span><span>control __CTRL_SHA__</span>
  <span>treatment __TREAT_SHA__</span>
  <span>__NCLIPS__ clips</span><span>__MODELS__</span></div>

  __CONCLUSION__

  __METRICS__

  __GROUP_RADARS__

  __VISUAL_EVIDENCE__

  __DIAGNOSTIC_EVIDENCE__

  <section>
    <h2>Reproduce</h2>
    <pre>python tools/sbsbench/run_eval.py --label ctrl                              # control (gates vs baselines)
python tools/sbsbench/run_eval.py --label treat --extra __TREAT_ARGS__     # treatment
python tools/sbsbench/build_report.py &lt;build&gt;/sbs_eval/ctrl &lt;build&gt;/sbs_eval/treat report.html</pre>
    <p style="color:var(--muted);font-size:13px;margin-top:12px">Metrics: <code>tools/sbsbench/sbsbench.py</code>
    &middot; gate: <code>thresholds.json</code> &middot; plan: <code>docs/sbs-benchmark-plan.md</code></p>
  </section>

  <details class="fold bar-fold">
    <summary>Per-clip bar comparison — __CTRL_NAME__ → __TREAT_NAME__</summary>
    <div class="fold-body"><p class="sub">Matched <b>__CTRL_TAG__</b> and <b>__TREAT_TAG__</b> bars
    for every clip. Bars share a scale within each metric; exact values remain printed beside them.
    Green is better, red is worse, and grey is within noise.</p>
    __CHARTS__
    __FOOTER__</div>
  </details>
</div>
"""

models = ", ".join(sorted({m for r in (CTRL, TREAT)
                           for m in {e["meta"].get("model", "?") for e in r["clips"].values()}}))
if IS_MODE_CMP:
    h1 = f"{CTRL_NAME} vs. {TREAT_NAME}"
    lede = (f"Comparing two pipeline modes on identical clips: <b>{CTRL_NAME}</b> against "
            f"<b>{TREAT_NAME}</b>. Neither is a regression of the other — it is a tradeoff, "
            f"read from the per-metric split and the per-clip evidence below.")
else:
    h1 = "Control vs. treatment, by issue"
    lede = (f"Matched comparison-only run: <b>{CTRL_NAME}</b> against <b>{TREAT_NAME}</b>; "
            "committed baselines were not consulted." if IS_COMPARISON_ONLY else
            f"Treatment under test: <b>{TREAT_NAME}</b>, gated against the committed baselines.")
ctrl_sha = CTRL["meta"].get("git_sha", "?") + ("+dirty" if CTRL["meta"].get("git_dirty") else "")
treat_sha = TREAT["meta"].get("git_sha", "?") + ("+dirty" if TREAT["meta"].get("git_dirty") else "")
HTML = (HTML.replace("__H1__", h1).replace("__LEDE__", lede)
        .replace("__CTRL_NAME__", CTRL_NAME).replace("__TREAT_NAME__", TREAT_NAME)
        .replace("__DATE__", meta["timestamp"][:10]).replace("__CTRL_SHA__", ctrl_sha)
        .replace("__TREAT_SHA__", treat_sha)
        .replace("__NCLIPS__", str(len(CLIPS)))
        .replace("__MODELS__", models).replace("__CONCLUSION__", conclusion_section())
        .replace("__VISUAL_EVIDENCE__", visual_evidence_section())
        .replace("__DIAGNOSTIC_EVIDENCE__", diagnostic_evidence_section())
        .replace("__CTRL_TAG__", CTRL_TAG).replace("__TREAT_TAG__", TREAT_TAG)
        .replace("__GROUP_RADARS__", grouped_quality_section())
        .replace("__CHARTS__", scorecard_charts())
        .replace("__METRICS__", metrics_section())
        .replace("__FOOTER__", clean_footer())
        .replace("__TREAT_ARGS__", " ".join(TREAT["meta"].get("extra_args") or ["--mode game"])))
os.makedirs(os.path.dirname(os.path.abspath(out_html)), exist_ok=True)
with open(out_html, "w", encoding="utf-8") as f:
    f.write(HTML)
print("wrote", out_html, f"({len(HTML) // 1024} KB)")
