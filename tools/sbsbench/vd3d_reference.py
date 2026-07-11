#!/usr/bin/env python3
"""Prepare and score a local VisionDepth3D Bestv2 reference without committing its media.

The comparison is intentionally split into independent checkpoints:
  raw model output  Apollo raw_*.f32 vs VD3D raw_*.f32 (when exported)
  warp input depth  Apollo depth_*.png vs VD3D depth_*.png (when exported)
  final SBS         Apollo sbs_*.png vs the VD3D Bestv2 render

Final-SBS similarity does not decide which warp is better. It only measures reproduction. Artifact
and performance metrics decide the later Apollo-probe vs VD3D-style warp A/B.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import imageio.v3 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image

import sbsbench


# These are computable on both final SBS streams without borrowing Apollo's depth map. Metrics
# such as edge accuracy, stretch/rim masks, and the combined score are not comparable when the
# VD3D render has no matching depth checkpoint, so reporting them would reward missing inputs.
FINAL_SBS_COMPARABLE = {
    "pop_px_p50", "pop_px_p95", "pop_pct_p50", "pop_spread_px", "pop_spread_pct",
    "vmisalign_px", "flicker_p50", "flicker_p95",
}


def fail(message):
    print("vd3d_reference: " + message, file=sys.stderr)
    raise SystemExit(2)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_ffmpeg(args):
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error"] + args
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode:
        fail("ffmpeg failed: " + (r.stderr or r.stdout)[-2000:])


def alignment_error(source_by_id, ref_by_id, offset):
    vals = []
    ids = sorted(source_by_id)
    ref_ids = sorted(ref_by_id)
    for source_pos, frame_id in enumerate(ids):
        target_pos = source_pos + offset
        if not 0 <= target_pos < len(ref_ids):
            continue
        src = Image.open(source_by_id[frame_id]).convert("RGB").resize((320, 180), Image.BILINEAR)
        ref = Image.open(ref_by_id[ref_ids[target_pos]]).convert("RGB")
        w = ref.width // 2
        left = ref.crop((0, 0, w, ref.height)).resize((320, 180), Image.BILINEAR)
        right = ref.crop((w, 0, 2 * w, ref.height)).resize((320, 180), Image.BILINEAR)
        a = np.asarray(src, np.float32) / 255.0
        midpoint = 0.5 * (np.asarray(left, np.float32) + np.asarray(right, np.float32)) / 255.0
        vals.append(float(np.mean(np.abs(a - midpoint))))
    return float(np.mean(vals)) if vals else float("inf")


def prepare(args):
    os.makedirs(args.out, exist_ok=True)
    source_dir = os.path.join(args.out, "source")
    source_all_dir = os.path.join(args.out, "source_all")
    reference_dir = os.path.join(args.out, "bestv2")
    bootstrap_dir = os.path.join(args.out, "bootstrap")
    shutil.rmtree(source_dir, ignore_errors=True)
    shutil.rmtree(source_all_dir, ignore_errors=True)
    shutil.rmtree(reference_dir, ignore_errors=True)
    shutil.rmtree(bootstrap_dir, ignore_errors=True)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(source_all_dir, exist_ok=True)
    os.makedirs(reference_dir, exist_ok=True)
    os.makedirs(bootstrap_dir, exist_ok=True)

    src_meta = iio.immeta(args.source_video)
    ref_meta = iio.immeta(args.vd3d_sbs)
    ref_w, ref_h = map(int, ref_meta["size"])
    if ref_w % 2:
        fail(f"reference SBS width must be even, got {ref_w}")
    if abs(float(src_meta["fps"]) - float(ref_meta["fps"])) > 1e-6:
        fail(f"FPS mismatch: source={src_meta['fps']} reference={ref_meta['fps']}")
    eye_w = ref_w // 2
    total_frames, _ = imageio_ffmpeg.count_frames_and_secs(args.source_video)
    bootstrap_count = min(5, max(3, total_frames // 300))
    bootstrap_indices = np.linspace(0, total_frames - 1, bootstrap_count, dtype=int).tolist()

    select = f"select='not(mod(n,{args.stride}))'"
    # Keep every native source frame. The harness processes all of them so temporal state is real,
    # then --output-every selects comparable artifacts without changing that state.
    run_ffmpeg(["-y", "-i", args.source_video, "-start_number", "0",
                os.path.join(source_all_dir, "frame_%05d.png")])
    all_source = sbsbench.indexed_files(os.path.join(source_all_dir, "frame_*.png"), "frame_")
    sampled_original_ids = [i for i in sorted(all_source) if i % args.stride == 0]
    for frame_id in sampled_original_ids:
        shutil.copy2(all_source[frame_id], os.path.join(source_dir, f"frame_{frame_id:05d}.png"))
    run_ffmpeg(["-y", "-i", args.vd3d_sbs, "-vf", select, "-fps_mode", "vfr",
                "-start_number", "0", os.path.join(reference_dir, "sbs_%05d.png")])
    # ffmpeg numbers selected outputs densely; restore their ORIGINAL source identities.
    dense_reference = sbsbench.indexed_files(os.path.join(reference_dir, "sbs_*.png"), "sbs_")
    if len(dense_reference) != len(sampled_original_ids):
        fail(f"selected reference count {len(dense_reference)} != source count {len(sampled_original_ids)}")
    staged = []
    for dense_id, original_id in zip(sorted(dense_reference), sampled_original_ids):
        temporary = os.path.join(reference_dir, f"tmp_{original_id:05d}.png")
        os.replace(dense_reference[dense_id], temporary)
        staged.append((temporary, os.path.join(reference_dir, f"sbs_{original_id:05d}.png")))
    for temporary, final in staged:
        os.replace(temporary, final)
    for frame_index in bootstrap_indices:
        shutil.copy2(all_source[frame_index],
                     os.path.join(bootstrap_dir, f"frame_{frame_index:05d}.png"))

    source = sbsbench.indexed_files(os.path.join(source_dir, "frame_*.png"), "frame_")
    reference = sbsbench.indexed_files(os.path.join(reference_dir, "sbs_*.png"), "sbs_")
    if set(source) != set(reference):
        fail(f"extracted frame identities differ: source={sorted(source)} reference={sorted(reference)}")
    errors = {str(offset): alignment_error(source, reference, offset) for offset in range(-2, 3)}
    best_offset = min(errors, key=errors.get)
    if best_offset != "0":
        fail(f"reference is not frame-aligned; best offset={best_offset}, errors={errors}")
    nonzero = min(v for k, v in errors.items() if k != "0")
    if errors["0"] >= nonzero * 0.9:
        fail(f"alignment is ambiguous: errors={errors}")

    manifest = {
        "schema": 1,
        "source_video": os.path.abspath(args.source_video),
        "source_sha256": file_sha256(args.source_video),
        "vd3d_sbs": os.path.abspath(args.vd3d_sbs),
        "vd3d_sbs_sha256": file_sha256(args.vd3d_sbs),
        "preset": "Bestv2",
        "fps": float(src_meta["fps"]),
        "source_size": list(map(int, src_meta["size"])),
        "reference_size": [ref_w, ref_h],
        "eye_size": [eye_w, ref_h],
        "source_total_frames": total_frames,
        "bootstrap_indices": bootstrap_indices,
        "stride": args.stride,
        "frame_ids": sorted(source),
        "source_frame_index_by_id": {str(i): i for i in sorted(source)},
        "alignment_mae_by_offset": errors,
    }
    with open(os.path.join(args.out, "reference_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"prepared {len(source)} aligned frames in {args.out}")


def corr(a, b):
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    den = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / den) if den else 0.0


def resize_float(a, width, height):
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize(
        (width, height), Image.BILINEAR), dtype=np.float32)


def load_raw_dir(path):
    shape_path = os.path.join(path, "raw_shape.json")
    if not os.path.exists(shape_path):
        fail(f"missing {shape_path}")
    shape = json.load(open(shape_path))
    w, h = int(shape["width"]), int(shape["height"])
    files = sbsbench.indexed_files(os.path.join(path, "raw_*.f32"), "raw_")
    arrays = {}
    for frame_id, file_path in files.items():
        a = np.fromfile(file_path, dtype="<f4")
        if a.size != w * h:
            fail(f"{file_path}: expected {w * h} floats, got {a.size}")
        arrays[frame_id] = a.reshape(h, w)
    return arrays


def compare_raw(apollo_dir, vd3d_dir):
    a = load_raw_dir(apollo_dir)
    v = load_raw_dir(vd3d_dir)
    if set(a) != set(v):
        fail("raw checkpoint frame identities differ")
    rows = []
    for frame_id in sorted(a):
        aa = resize_float(a[frame_id], v[frame_id].shape[1], v[frame_id].shape[0])
        vv = v[frame_id]
        rows.append({"frame": frame_id, "corr": corr(aa, vv), "corr_inverted": corr(-aa, vv)})
    return {"frames": rows, "corr_mean": float(np.mean([r["corr"] for r in rows])),
            "corr_min": float(np.min([r["corr"] for r in rows]))}


def compare_warp_depth(apollo_dir, vd3d_dir):
    a = sbsbench.indexed_files(os.path.join(apollo_dir, "depth_*.png"), "depth_")
    v = sbsbench.indexed_files(os.path.join(vd3d_dir, "depth_*.png"), "depth_")
    if set(a) != set(v):
        fail("warp-input depth checkpoint frame identities differ")
    rows = []
    for frame_id in sorted(a):
        aa = sbsbench.load_depth(a[frame_id])
        vv = sbsbench.load_depth(v[frame_id])
        aa = resize_float(aa, vv.shape[1], vv.shape[0])
        # Apollo stores high=near; VD3D inverts its depth video before rendering (low=near).
        physical = 1.0 - aa
        rows.append({"frame": frame_id, "stored_corr": corr(aa, vv),
                     "physical_corr": corr(physical, vv),
                     "physical_mae": float(np.mean(np.abs(physical - vv)))})
    return {"frames": rows,
            "physical_corr_mean": float(np.mean([r["physical_corr"] for r in rows])),
            "physical_corr_min": float(np.min([r["physical_corr"] for r in rows])),
            "physical_mae_mean": float(np.mean([r["physical_mae"] for r in rows])),
            "polarity": "Apollo high-near compared as 1-Apollo to VD3D low-near"}


def score(args):
    manifest = json.load(open(os.path.join(args.reference, "reference_manifest.json")))
    for key, hash_key in [("source_video", "source_sha256"), ("vd3d_sbs", "vd3d_sbs_sha256")]:
        path = manifest[key]
        if not os.path.exists(path) or file_sha256(path) != manifest[hash_key]:
            fail(f"reference input changed or is missing: {path}")
    source_dir = os.path.join(args.reference, "source")
    vd3d_sbs_dir = os.path.join(args.reference, "bestv2")
    apollo = sbsbench.measure_sequence(args.apollo_out, source_dir)
    vd3d = sbsbench.measure_sequence(vd3d_sbs_dir, source_dir)
    if not apollo or not vd3d:
        fail("missing Apollo or VD3D SBS sequence")
    _, aagg = apollo
    _, vagg = vd3d
    shared = sorted(FINAL_SBS_COMPARABLE & set(aagg) & set(vagg))
    final = {"apollo": {k: aagg[k] for k in shared}, "vd3d": {k: vagg[k] for k in shared},
             "apollo_minus_vd3d": {k: aagg[k] - vagg[k] for k in shared}}

    # Pixel similarity is a reproduction metric, not a quality verdict. A different warp can be
    # better while scoring worse here; the later dual-warp A/B uses artifact/perf gates.
    ap = sbsbench.indexed_files(os.path.join(args.apollo_out, "sbs_*.png"), "sbs_")
    vp = sbsbench.indexed_files(os.path.join(vd3d_sbs_dir, "sbs_*.png"), "sbs_")
    if set(ap) != set(vp):
        fail("Apollo/VD3D final SBS frame identities differ")
    pixel_rows = []
    for frame_id in sorted(ap):
        aa = sbsbench.load_rgb(ap[frame_id])
        vv = sbsbench.load_rgb(vp[frame_id])
        if aa.shape != vv.shape:
            fail(f"frame {frame_id}: final SBS shape mismatch {aa.shape} vs {vv.shape}")
        mse = float(np.mean((aa - vv) ** 2))
        pixel_rows.append({"frame": frame_id, "mae": float(np.mean(np.abs(aa - vv))),
                           "psnr": float(-10.0 * np.log10(max(mse, 1e-12)))})
    final["pixel_reproduction"] = {
        "mae_mean": float(np.mean([r["mae"] for r in pixel_rows])),
        "psnr_mean": float(np.mean([r["psnr"] for r in pixel_rows])),
        "frames": pixel_rows,
    }
    out = {"manifest": manifest, "final_sbs_reproduction": final}
    if args.vd3d_raw:
        out["raw_model_checkpoint"] = compare_raw(args.apollo_out, args.vd3d_raw)
    if args.vd3d_warp_depth:
        out["warp_input_depth_checkpoint"] = compare_warp_depth(args.apollo_out, args.vd3d_warp_depth)
    gates = {}
    if "raw_model_checkpoint" in out:
        raw = out["raw_model_checkpoint"]
        gates["raw_model_corr_min"] = {"value": raw["corr_min"], "required": 0.995,
                                       "pass": raw["corr_min"] >= 0.995}
    if "warp_input_depth_checkpoint" in out:
        depth = out["warp_input_depth_checkpoint"]
        gates["warp_depth_corr_min"] = {"value": depth["physical_corr_min"], "required": 0.995,
                                        "pass": depth["physical_corr_min"] >= 0.995}
        gates["warp_depth_mae_mean"] = {"value": depth["physical_mae_mean"], "required": 0.04,
                                        "pass": depth["physical_mae_mean"] <= 0.04}
    out["depth_stage_gates"] = gates
    out["depth_stage_verdict"] = "pass" if gates and all(g["pass"] for g in gates.values()) else (
        "not-run" if not gates else "fail")
    output = args.json or os.path.join(args.apollo_out, "vd3d_comparison.json")
    with open(output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {output} (depth-stage verdict: {out['depth_stage_verdict']})")
    if out["depth_stage_verdict"] == "fail":
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="extract and verify a local aligned Bestv2 reference")
    p.add_argument("--source-video", required=True)
    p.add_argument("--vd3d-sbs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=8)
    p.set_defaults(func=prepare)
    s = sub.add_parser("score", help="compare Apollo checkpoints/output to the local reference")
    s.add_argument("--reference", required=True)
    s.add_argument("--apollo-out", required=True)
    s.add_argument("--vd3d-raw", help="optional VD3D raw_*.f32 checkpoint directory")
    s.add_argument("--vd3d-warp-depth", help="optional VD3D depth_*.png checkpoint directory")
    s.add_argument("--json")
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
