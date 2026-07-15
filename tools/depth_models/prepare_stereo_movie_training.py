#!/usr/bin/env python3
"""Split a stereoscopic movie into complete-shot artistic-policy clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def change_score(previous, current):
    return float(np.mean(np.abs(
        previous.astype(np.float32) - current.astype(np.float32)
    )) / 255.0)


def analysis_gray(eye):
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)


def split_eyes(frame, layout, eye_order):
    height, width = frame.shape[:2]
    if layout == "above-below":
        if height % 2:
            raise RuntimeError(f"above/below frame has odd height: {height}")
        first, second = frame[:height // 2], frame[height // 2:]
    else:
        if width % 2:
            raise RuntimeError(f"side-by-side frame has odd width: {width}")
        first, second = frame[:, :width // 2], frame[:, width // 2:]
    return (first, second) if eye_order == "first-left" else (second, first)


def prepare(video: Path, output: Path, name: str, domain: str,
            layout="above-below", eye_order="first-left", sample_fps=2.0,
            cut_threshold=0.18, output_width=1920, start_seconds=0.0,
            end_margin_seconds=0.0, split="training", film_id="",
            homepage="", license_name="user-provided; verify rights",
            policy_role="cinematic_training", eye_aspect_ratio=0.0,
            global_policy_weight=1.0):
    if global_policy_weight <= 0.0:
        raise RuntimeError("global policy weight must be positive")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stereo movie: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0.0 or source_frames <= 0:
        raise RuntimeError("movie has invalid frame rate or frame count")
    stride = max(1, round(source_fps / sample_fps))
    start_frame = max(0, round(start_seconds * source_fps))
    end_frame = source_frames - max(0, round(end_margin_seconds * source_fps))
    if start_frame >= end_frame:
        raise RuntimeError("movie trim removes all frames")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    rows = []
    shot_index = -1
    shot_frame = 0
    previous_gray = None
    frame_index = start_frame
    while frame_index < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        left, right = split_eyes(frame, layout, eye_order)
        current_gray = analysis_gray(left)
        score = (0.0 if previous_gray is None
                 else change_score(previous_gray, current_gray))
        if previous_gray is None or score >= cut_threshold:
            shot_index += 1
            shot_frame = 0
            rows.append({
                "shot": shot_index,
                "clip": f"{domain}_shot_{shot_index:04d}",
                "source_start_frame": frame_index,
                "samples": 0,
                "context_frames": 0,
                "max_change_score": score,
            })
        else:
            rows[-1]["max_change_score"] = max(
                rows[-1]["max_change_score"], score
            )
        previous_gray = current_gray

        stored_aspect = left.shape[1] / left.shape[0]
        aspect_differs = (
            eye_aspect_ratio > 0.0
            and abs(stored_aspect - eye_aspect_ratio) > 1e-6
        )
        if output_width > 0 and (
            left.shape[1] != output_width or aspect_differs
        ):
            output_height = (
                round(output_width / eye_aspect_ratio)
                if eye_aspect_ratio > 0.0
                else round(left.shape[0] * output_width / left.shape[1])
            )
            size = (output_width, output_height)
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA)
        clip = output / f"{domain}_shot_{shot_index:04d}"
        frame_name = f"frame_{shot_frame:05d}.png"
        clip.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(clip / frame_name), left):
            raise RuntimeError(f"cannot write {clip / frame_name}")
        rows[-1]["context_frames"] += 1
        # Runtime latches the global policy on the first resolved frame after a cut. Always
        # supervise that exact shot identity; a production-global sampling phase can otherwise
        # leave the latch frame unseen even when later frames from the shot are labelled.
        if shot_frame == 0 or (frame_index - start_frame) % stride == 0:
            (clip / "gt_right").mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(clip / "gt_right" / frame_name), right):
                raise RuntimeError(f"cannot write {clip / 'gt_right' / frame_name}")
            rows[-1]["samples"] += 1
        rows[-1]["source_end_frame"] = frame_index
        shot_frame += 1
        frame_index += 1
    capture.release()
    if not rows:
        raise RuntimeError("movie preparation produced no shots")

    for row in rows:
        row["split"] = split
        clip = output / f"{domain}_shot_{row['shot']:04d}"
        meta = {
            "schema": 1,
            "name": f"{name}-shot-{row['shot']:04d}",
            "dataset": name,
            "domain": domain,
            "film_id": film_id or domain,
            "homepage": homepage or None,
            "license": license_name,
            "purpose": "artistic-policy authored stereoscopic movie supervision",
            "policy_role": policy_role,
            "global_policy_weight": global_policy_weight,
            "shot": row["shot"],
            "source_start_frame": row["source_start_frame"],
            "source_end_frame": row["source_end_frame"],
            "sampling_fps": source_fps / stride,
            "label_sampling": "shot-first-frame-plus-production-cadence",
            "context_fps": source_fps,
            "display_eye_aspect_ratio": (
                eye_aspect_ratio if eye_aspect_ratio > 0.0
                else left.shape[1] / left.shape[0]
            ),
            "split": split,
            "required_gt_stereo": True,
        }
        (clip / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "schema": 1,
        "dataset": name,
        "domain": domain,
        "policy_role": policy_role,
        "film_id": film_id or domain,
        "homepage": homepage or None,
        "license": license_name,
        "split": split,
        "video": str(video.resolve()),
        "video_size": video.stat().st_size,
        "video_sha256": sha256(video),
        "layout": layout,
        "eye_order": eye_order,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "start_seconds": start_seconds,
        "end_margin_seconds": end_margin_seconds,
        "sample_stride": stride,
        "sampling_fps": source_fps / stride,
        "label_sampling": "shot-first-frame-plus-production-cadence",
        "cut_threshold": cut_threshold,
        "output_width": output_width,
        "display_eye_aspect_ratio": eye_aspect_ratio or None,
        "global_policy_weight": global_policy_weight,
        "shots": rows,
        "sequences": [
            {
                "clip": row["clip"],
                "source_sequence": f"shot_{row['shot']:04d}",
                "frames": row["context_frames"],
                "label_frames": row["samples"],
                "split": row["split"],
            }
            for row in rows
        ],
        "shot_count": len(rows),
        "sample_count": sum(row["samples"] for row in rows),
        "context_frame_count": sum(row["context_frames"] for row in rows),
        "training_samples": sum(
            row["samples"] for row in rows if row["split"] == "training"
        ),
        "development_samples": sum(
            row["samples"] for row in rows if row["split"] == "development"
        ),
        "test_samples": sum(
            row["samples"] for row in rows if row["split"] == "test"
        ),
        "split_rule": "the complete film has one split; never split its shots",
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="Big Buck Bunny stereoscopic re-render")
    parser.add_argument("--domain", default="big_buck_bunny_3d")
    parser.add_argument("--layout", choices=("above-below", "side-by-side"),
                        default="above-below")
    parser.add_argument("--eye-order", choices=("first-left", "first-right"),
                        default="first-left")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--cut-threshold", type=float, default=0.18)
    parser.add_argument("--output-width", type=int, default=1920)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-margin-seconds", type=float, default=0.0)
    parser.add_argument("--split", choices=("training", "development", "test"),
                        default="training")
    parser.add_argument("--film-id", default="")
    parser.add_argument("--homepage", default="")
    parser.add_argument("--license", default="user-provided; verify rights")
    parser.add_argument("--policy-role", default="cinematic_training")
    parser.add_argument(
        "--eye-aspect-ratio", type=float, default=0.0,
        help="restore an anamorphically packed eye to this display aspect",
    )
    parser.add_argument("--global-policy-weight", type=float, default=1.0)
    args = parser.parse_args()
    manifest = prepare(
        args.video, args.output, args.name, args.domain, args.layout,
        args.eye_order, args.sample_fps, args.cut_threshold, args.output_width,
        args.start_seconds, args.end_margin_seconds, args.split, args.film_id,
        args.homepage, args.license, args.policy_role,
        args.eye_aspect_ratio, args.global_policy_weight,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
