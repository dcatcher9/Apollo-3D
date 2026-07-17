#!/usr/bin/env python3
"""Split a stereoscopic movie into complete-shot artistic-policy clips."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import video_color_contract as video_color


_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def _validate_domain(domain):
    """Return a domain that is safe to embed in a clip directory name."""
    if (not isinstance(domain, str) or not domain or
            domain != domain.strip() or domain in {".", ".."} or
            any(character in '<>:"/\\|?*' or ord(character) < 32
                for character in domain) or domain.endswith((".", " ")) or
            domain.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES):
        raise RuntimeError(
            "domain must be a safe non-reserved single path component"
        )
    candidate = Path(domain)
    if candidate.is_absolute() or candidate.name != domain or len(candidate.parts) != 1:
        raise RuntimeError("domain must be a safe single path component")
    return domain


def _assert_direct_child(path: Path, output: Path):
    """Reject traversal and existing links that escape the prepared root."""
    try:
        output_root = output.resolve(strict=False)
        resolved = path.resolve(strict=False)
        parent = path.parent.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise RuntimeError(f"cannot validate prepared path: {path}") from error
    if parent != output_root or resolved.parent != output_root:
        raise RuntimeError(f"refusing prepared path outside output: {path}")


class _BoundedPngWriter:
    """Bounded asynchronous PNG writer with a synchronous single-worker mode."""

    def __init__(self, workers):
        if workers < 1:
            raise RuntimeError("write_workers must be at least 1")
        self.max_pending = 2 * workers
        self.pending = deque()
        self.peak_pending = 0
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="png-write")
            if workers > 1 else None
        )

    @staticmethod
    def _write(path, image):
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"cannot write {path}")

    def submit(self, path, image):
        if self.executor is None:
            self._write(path, image)
            return
        while len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(self._write, path, image))
        self.peak_pending = max(self.peak_pending, len(self.pending))

    def wait(self):
        while self.pending:
            self.pending.popleft().result()

    def close(self):
        self.wait()
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def abort(self):
        for future in self.pending:
            future.cancel()
        self.pending.clear()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


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
            global_policy_weight=1.0, ffprobe=None, input_color="auto",
            write_workers=4):
    domain = _validate_domain(domain)
    if sample_fps <= 0.0:
        raise RuntimeError("sample_fps must be positive")
    if layout not in {"above-below", "side-by-side"}:
        raise RuntimeError(f"unsupported stereo layout: {layout}")
    if eye_order not in {"first-left", "first-right"}:
        raise RuntimeError(f"unsupported stereo eye order: {eye_order}")
    if global_policy_weight <= 0.0:
        raise RuntimeError("global policy weight must be positive")
    if write_workers < 1:
        raise RuntimeError("write_workers must be at least 1")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    color_probe = video_color.probe_sdr_input(video, ffprobe, input_color)
    video_size = video.stat().st_size
    video_sha256 = sha256(video)
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        shutil.rmtree(output)
        raise RuntimeError(f"cannot open stereo movie: {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0.0 or source_frames <= 0:
        capture.release()
        shutil.rmtree(output)
        raise RuntimeError("movie has invalid frame rate or frame count")
    stride = max(1, round(source_fps / sample_fps))
    start_frame = max(0, round(start_seconds * source_fps))
    end_frame = source_frames - max(0, round(end_margin_seconds * source_fps))
    if start_frame >= end_frame:
        capture.release()
        shutil.rmtree(output)
        raise RuntimeError("movie trim removes all frames")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = _BoundedPngWriter(write_workers)
    rows = []
    dropped_shots = []
    shot_index = -1
    shot = None
    previous_gray = None
    frame_index = start_frame

    def begin_shot(source_index, transition_score):
        nonlocal shot_index
        shot_index += 1
        clip_name = f"{domain}_shot_{shot_index:04d}"
        partial = output / f".{clip_name}.partial"
        _assert_direct_child(partial, output)
        if partial.exists():
            shutil.rmtree(partial)
        _assert_direct_child(partial, output)
        partial.mkdir()
        return {
            "shot": shot_index,
            "clip": clip_name,
            "partial": partial,
            "source_start_frame": source_index,
            "samples": 0,
            "label_frame_ids": [],
            "context_frames": 0,
            "max_change_score": transition_score,
        }

    def append_frame(state, source_index, left, right):
        shot_frame = state["context_frames"]
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
        display_aspect = left.shape[1] / left.shape[0]
        if "display_eye_aspect_ratio" in state:
            if abs(state["display_eye_aspect_ratio"] - display_aspect) > 1e-6:
                raise RuntimeError("movie resolution changed inside a detected shot")
        else:
            state["display_eye_aspect_ratio"] = display_aspect
        frame_name = f"frame_{shot_frame:05d}.png"
        writer.submit(state["partial"] / frame_name, left)
        state["context_frames"] += 1
        # Runtime latches the global policy on the first resolved frame after a cut. Always
        # supervise that exact shot identity; a production-global sampling phase can otherwise
        # leave the latch frame unseen even when later frames from the shot are labelled.
        if shot_frame == 0 or (source_index - start_frame) % stride == 0:
            right_dir = state["partial"] / "gt_right"
            right_dir.mkdir(parents=True, exist_ok=True)
            writer.submit(right_dir / frame_name, right)
            state["samples"] += 1
            state["label_frame_ids"].append(shot_frame)
        state["source_end_frame"] = source_index

    def finish_shot(state):
        if state is None:
            return
        # Never publish a clip while any frame write can still fail.
        writer.wait()
        if state["context_frames"] < 2:
            dropped_shots.append({
                "shot": state["shot"],
                "source_start_frame": state["source_start_frame"],
                "source_end_frame": state["source_end_frame"],
                "context_frames": state["context_frames"],
                "reason": "insufficient-temporal-context",
            })
            _assert_direct_child(state["partial"], output)
            shutil.rmtree(state["partial"])
            return
        state["split"] = split
        (state["partial"] / "label_frames.json").write_text(
            json.dumps({
                "schema": 1,
                "frame_ids": state["label_frame_ids"],
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        meta = {
            "schema": 1,
            "name": f"{name}-shot-{state['shot']:04d}",
            "dataset": name,
            "domain": domain,
            "film_id": film_id or domain,
            "homepage": homepage or None,
            "license": license_name,
            "purpose": "artistic-policy authored stereoscopic movie supervision",
            "policy_role": policy_role,
            "global_policy_weight": global_policy_weight,
            "shot": state["shot"],
            "source_start_frame": state["source_start_frame"],
            "source_end_frame": state["source_end_frame"],
            "sampling_fps": source_fps / stride,
            "label_sampling": "shot-first-frame-plus-production-cadence",
            "label_frame_count": len(state["label_frame_ids"]),
            "source_kind": "authored-stereo",
            "temporal_contract": "full-cadence-shot",
            "context_fps": source_fps,
            "display_eye_aspect_ratio": (
                eye_aspect_ratio if eye_aspect_ratio > 0.0
                else state["display_eye_aspect_ratio"]
            ),
            "split": split,
            "required_gt_stereo": True,
            "color_contract": color_probe["dataset_color_contract"],
            "source_color_probe": color_probe,
            "write_worker_count": write_workers,
        }
        (state["partial"] / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        clip = output / state["clip"]
        _assert_direct_child(clip, output)
        if clip.exists():
            raise RuntimeError(f"refusing to replace prepared clip: {clip}")
        _assert_direct_child(state["partial"], output)
        _assert_direct_child(clip, output)
        state["partial"].rename(clip)
        state.pop("partial")
        state.pop("display_eye_aspect_ratio")
        rows.append(state)

    try:
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            left, right = split_eyes(frame, layout, eye_order)
            current_gray = analysis_gray(left)
            score = (0.0 if previous_gray is None
                     else change_score(previous_gray, current_gray))
            if previous_gray is None or score >= cut_threshold:
                finish_shot(shot)
                shot = begin_shot(frame_index, score)
            else:
                shot["max_change_score"] = max(
                    shot["max_change_score"], score
                )
            previous_gray = current_gray
            append_frame(shot, frame_index, left, right)
            frame_index += 1
        finish_shot(shot)
        writer.close()
    except Exception:
        writer.abort()
        if output.exists():
            shutil.rmtree(output)
        raise
    finally:
        capture.release()

    if not rows:
        shutil.rmtree(output)
        raise RuntimeError("movie preparation produced no temporally valid shots")

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
        "video_size": video_size,
        "video_sha256": video_sha256,
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
        "color_contract": color_probe["dataset_color_contract"],
        "source_color_probe": color_probe,
        "write_worker_count": write_workers,
        "shots": rows,
        "dropped_shots": dropped_shots,
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
        "dropped_shot_count": len(dropped_shots),
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
    try:
        (output / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
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
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument(
        "--input-color", choices=("auto", "sdr"), default="auto",
        help="auto rejects HDR/high-bit-depth input; sdr records a reviewed SDR override",
    )
    args = parser.parse_args()
    manifest = prepare(
        args.video, args.output, args.name, args.domain, args.layout,
        args.eye_order, args.sample_fps, args.cut_threshold, args.output_width,
        args.start_seconds, args.end_margin_seconds, args.split, args.film_id,
        args.homepage, args.license, args.policy_role,
        args.eye_aspect_ratio, args.global_policy_weight, args.ffprobe,
        args.input_color, args.write_workers,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
