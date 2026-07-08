#!/usr/bin/env python3
"""Crop matched control/treatment left-eye regions at the strongest depth silhouette of a
harness frame, for the visual report. Also emits a depth crop with the disocclusion band marked."""
import os
import sys

import numpy as np
from PIL import Image

ctrl, treat, idx, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)
n = f"{idx:05d}"


def gray(p):
    return np.asarray(Image.open(p).convert("L"), np.float32)


depth = gray(os.path.join(ctrl, f"depth_{n}.png"))
sbs_c = Image.open(os.path.join(ctrl, f"sbs_{n}.png")).convert("RGB")
sbs_t = Image.open(os.path.join(treat, f"sbs_{n}.png")).convert("RGB")
W, H = sbs_c.size
ew, eh = W // 2, H
dh, dw = depth.shape

# Strongest vertical silhouette in the central region (left eye disocclusion site).
gx = np.abs(np.diff(depth, axis=1, prepend=depth[:, :1]))
band = gx[int(dh * 0.15):int(dh * 0.85)]
colscore = band.sum(0)
lo, hi = int(dw * 0.15), int(dw * 0.85)
cx_d = int(np.argmax(colscore[lo:hi]) + lo)
rowscore = gx[:, max(0, cx_d - 2):cx_d + 3].sum(1)
cy_d = int(np.argmax(rowscore))
cx, cy = int(cx_d / dw * ew), int(cy_d / dh * eh)

cw, ch = 480, 360
x0 = max(0, min(ew - cw, cx - cw // 2))
y0 = max(0, min(eh - ch, cy - ch // 2))
print(f"silhouette at eye ({cx},{cy}) -> crop ({x0},{y0}) {cw}x{ch}")

for img, name in [(sbs_c, "control"), (sbs_t, "treat")]:
    left = img.crop((0, 0, ew, eh))
    crop = left.crop((x0, y0, x0 + cw, y0 + ch)).resize((cw * 2, ch * 2), Image.NEAREST)
    crop.save(os.path.join(outdir, f"crop_{name}.png"))

# Depth crop (upsampled to eye) with the silhouette column marked, so the reader sees the site.
du = np.asarray(Image.fromarray(depth.astype(np.uint8)).resize((ew, eh), Image.BILINEAR))
dcrop = np.stack([du] * 3, -1).astype(np.uint8)[y0:y0 + ch, x0:x0 + cw].copy()
mx = cx - x0
if 0 <= mx < cw:
    dcrop[:, max(0, mx - 1):mx + 2] = [255, 60, 60]
Image.fromarray(dcrop).resize((cw * 2, ch * 2), Image.NEAREST).save(os.path.join(outdir, "crop_depth.png"))
print(f"wrote crop_control.png crop_treat.png crop_depth.png to {outdir}")
