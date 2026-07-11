#!/usr/bin/env python3
"""Export VD3D DA-V2 raw and pre-warp depth checkpoints for an aligned reference clip.

Run this with a Python environment containing torch/torchvision/safetensors. It imports VD3D's
installed DA-V2 model definition as a reference and does not modify or copy the VD3D repository.
The post-model math is independently expressed from the documented pipeline:
  sparse p2/p98 bootstrap lock -> 8-bit depth-video boundary -> invert -> pixel EMA 0.5
  -> per-frame p2/p98 bounds EMA (old weight 0.82).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def fail(message):
    print("export_vd3d_depth_reference: " + message, file=sys.stderr)
    raise SystemExit(2)


def load_model(vd3d_repo, weights):
    # dpt.py imports cv2 only for its optional infer_image helper. The reference exporter calls
    # forward() directly, so constants are sufficient and avoid mutating the evaluation venv.
    cv2 = types.ModuleType("cv2")
    cv2.INTER_NEAREST, cv2.INTER_CUBIC, cv2.INTER_AREA = 0, 2, 3
    sys.modules.setdefault("cv2", cv2)
    sys.path.insert(0, os.path.join(vd3d_repo, "core", "models"))
    from depth_anything_v2.dpt import DepthAnythingV2
    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
    state = load_file(weights, device="cpu")
    model.load_state_dict(state, strict=True)
    return model.eval()


def frame_stream(ffmpeg, video, width, height):
    p = subprocess.Popen([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", video,
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = width * height * 3
    try:
        while True:
            data = p.stdout.read(size)
            if not data:
                break
            if len(data) != size:
                fail(f"truncated ffmpeg frame: expected {size} bytes, got {len(data)}")
            yield np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    finally:
        if p.stdout:
            p.stdout.close()
        rc = p.wait()
        if rc:
            fail(f"ffmpeg decode exited {rc}")


@torch.inference_mode()
def infer(model, rgb, requested_w=768, requested_h=432):
    # VD3D first resizes the decoded frame to its UI inference size, then the DA-V2 adapter snaps
    # both axes upward to patch-14 dimensions. For 768x432 this is exactly 770x434.
    x = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    x = F.interpolate(x, size=(requested_h, requested_w), mode="area")
    h = max(14, math.ceil(requested_h / 14) * 14)
    w = max(14, math.ceil(requested_w / 14) * 14)
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return model((x - mean) / std)[0].float().cpu().numpy().astype("<f4", copy=False)


def save_gray16(path, a):
    Image.fromarray(np.round(np.clip(a, 0, 1) * 65535.0).astype(np.uint16)).save(path)


def percentile(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="directory made by vd3d_reference.py prepare")
    ap.add_argument("--vd3d-repo", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ref = json.load(open(os.path.join(args.reference, "reference_manifest.json")))
    source_video = ref["source_video"]
    src_w, src_h = map(int, ref["source_size"])
    total_frames = int(ref["source_total_frames"])
    bootstrap_indices = set(map(int, ref["bootstrap_indices"]))
    stride = int(ref["stride"])
    os.makedirs(args.out, exist_ok=True)

    model = load_model(args.vd3d_repo, args.weights)

    # Sparse whole-clip bootstrap: widest p2/p98 range among linspace-selected frames.
    lo = None
    hi = None
    for frame_index, rgb in enumerate(frame_stream(args.ffmpeg, source_video, src_w, src_h)):
        if frame_index not in bootstrap_indices:
            continue
        raw = infer(model, rgb)
        frame_lo, frame_hi = percentile(raw, 2), percentile(raw, 98)
        lo = frame_lo if lo is None else min(lo, frame_lo)
        hi = frame_hi if hi is None else max(hi, frame_hi)
    if lo is None or hi is None or hi - lo < 1e-6:
        fail("could not learn the VD3D bootstrap range")

    temporal = None
    bounds_lo = bounds_hi = None
    output_count = 0
    raw_h = raw_w = 0
    for frame_index, rgb in enumerate(frame_stream(args.ffmpeg, source_video, src_w, src_h)):
        if frame_index >= total_frames:
            break
        raw = infer(model, rgb)
        raw_h, raw_w = raw.shape

        # Depth-generation video boundary. VD3D renders from an 8-bit grayscale depth video.
        depthgen = np.clip((raw - lo) / (hi - lo + 1e-6), 0, 1)
        depthgen_u8 = np.round(depthgen * 255.0).astype(np.uint8)
        render_depth = 1.0 - depthgen_u8.astype(np.float32) / 255.0  # VD3D low=near convention
        temporal = render_depth.copy() if temporal is None else 0.5 * temporal + 0.5 * render_depth
        frame_lo, frame_hi = percentile(temporal, 2), percentile(temporal, 98)
        if bounds_lo is None:
            bounds_lo, bounds_hi = frame_lo, frame_hi
        else:
            bounds_lo = 0.82 * bounds_lo + 0.18 * frame_lo
            bounds_hi = 0.82 * bounds_hi + 0.18 * frame_hi
        warp_depth = np.clip((temporal - bounds_lo) / (bounds_hi - bounds_lo + 1e-6), 0, 1)

        if frame_index % stride == 0:
            raw.tofile(os.path.join(args.out, f"raw_{frame_index:05d}.f32"))
            save_gray16(os.path.join(args.out, f"depthgen_{frame_index:05d}.png"), depthgen)
            save_gray16(os.path.join(args.out, f"depth_{frame_index:05d}.png"), warp_depth)
            output_count += 1

    with open(os.path.join(args.out, "raw_shape.json"), "w") as fh:
        json.dump({"width": raw_w, "height": raw_h, "dtype": "float32-le",
                   "stage": "VD3D DA-V2 raw model output"}, fh, indent=2)
    with open(os.path.join(args.out, "depth_reference_meta.json"), "w") as fh:
        json.dump({"frames": output_count, "bootstrap_lo": lo, "bootstrap_hi": hi,
                   "bootstrap_indices": sorted(bootstrap_indices), "requested_size": [768, 432],
                   "actual_patch14_size": [raw_w, raw_h], "polarity": "low-near after invert",
                   "pipeline": "bootstrap-p2p98 -> u8 -> invert -> EMA0.5 -> p2p98-bounds-EMA0.82"},
                  fh, indent=2)
    print(f"wrote {output_count} raw + pre-warp VD3D depth checkpoints to {args.out}")


if __name__ == "__main__":
    main()
