#!/usr/bin/env python3
"""
split_video - turn a short video into a fixed frame sequence for the SBS benchmark clip set.

A benchmark clip is just a directory of frame PNGs; this produces one deterministically from a
video so every A/B run feeds the exact same input (see docs/sbs-benchmark-plan.md). Uses the
ffmpeg bundled with imageio-ffmpeg (no system ffmpeg needed), or a system ffmpeg if on PATH.

Usage:
  python split_video.py clip.mp4 -o E:/ApolloDev/sbs_bench/clips/movie_fs
  python split_video.py clip.mp4 -o OUT --fps 24 --start 12 --duration 5 --width 5120

  --fps N        sample N frames/sec (default: keep source rate)
  --start S      start S seconds in       --duration D   only D seconds
  --width W      scale to width W (keeps aspect; default: source resolution)
  --max N        stop after N frames
Frames are written as frame_%05d.png (top-to-bottom RGB), the layout sbsbench and the harness expect.
"""
import argparse
import os
import shutil
import subprocess
import sys


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        sys.exit("no ffmpeg found (install imageio-ffmpeg: python -m pip install imageio-ffmpeg)")


def main():
    ap = argparse.ArgumentParser(description="Split a video into a benchmark frame sequence.")
    ap.add_argument("video")
    ap.add_argument("-o", "--out", required=True, help="output frames directory")
    ap.add_argument("--fps", type=float, help="sample rate (default: source)")
    ap.add_argument("--start", type=float, help="start offset seconds")
    ap.add_argument("--duration", type=float, help="clip length seconds")
    ap.add_argument("--width", type=int, help="scale to this width (aspect preserved)")
    ap.add_argument("--max", type=int, help="max frames")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"no such video: {args.video}")
    os.makedirs(args.out, exist_ok=True)

    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if args.start is not None:
        cmd += ["-ss", str(args.start)]
    cmd += ["-i", args.video]
    if args.duration is not None:
        cmd += ["-t", str(args.duration)]
    vf = []
    if args.fps:
        vf.append(f"fps={args.fps}")
    if args.width:
        vf.append(f"scale={args.width}:-2")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if args.max:
        cmd += ["-frames:v", str(args.max)]
    cmd += [os.path.join(args.out, "frame_%05d.png")]

    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    n = len([f for f in os.listdir(args.out) if f.startswith("frame_") and f.endswith(".png")])
    print(f"wrote {n} frames to {args.out}")


if __name__ == "__main__":
    main()
