#!/usr/bin/env python3
"""Prepare complete-shot monocular video for artistic-policy augmentation.

Every decoded frame is retained so Apollo's temporal depth state sees the same
full-cadence sequence as production.  A sparse ``label_frames.json`` manifest
selects the frames that need expensive render artifacts and metric scoring.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import shutil
from pathlib import Path

import cv2

import video_color_contract as video_color
import preprocessing_artifact_cache as artifact_cache
import monocular_movie_raw_cache as raw_movie


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
        self.workers = workers
        self.max_pending = 2 * workers
        self.pending = deque()
        self.peak_pending = 0
        self.executor = (
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="png-write")
            if workers > 1 else None
        )

    @staticmethod
    def _write(path, image, params):
        if not cv2.imwrite(str(path), image, params):
            raise RuntimeError(f"cannot write {path}")

    def submit(self, path, image, params):
        if self.executor is None:
            self._write(path, image, params)
            return
        while len(self.pending) >= self.max_pending:
            self.pending.popleft().result()
        self.pending.append(self.executor.submit(
            self._write, path, image, params
        ))
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


# The cache-enabled and direct paths must execute one implementation.  These
# aliases preserve the preparer's public test surface without duplicating the
# byte-producing resize or the threshold-independent cut signature.
analysis_gray = raw_movie.analysis_gray
change_score = raw_movie.change_score
resized_frame = raw_movie.resized_frame


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(video: Path, output: Path, name: str, domain: str,
            sample_fps=2.0, cut_threshold=0.18, output_width=1280,
            start_seconds=0.0, end_margin_seconds=0.0, split="training",
            production_id="", homepage="",
            license_name="user-provided; verify rights",
            policy_role="cinematic_training", global_policy_weight=1.0,
            min_context_frames=3, png_compression=1, ffprobe=None,
            input_color="auto", write_workers=4, preprocess_cache=None,
            cache_observer=None):
    domain = _validate_domain(domain)
    code_paths = {
        "mono_preparer": Path(__file__).resolve(),
        "video_color_contract": Path(video_color.__file__).resolve(),
        "artifact_cache": Path(artifact_cache.__file__).resolve(),
    }
    code_identity = artifact_cache.code_identities(code_paths)
    raw_code_paths = raw_movie.code_paths()
    raw_code_identity = artifact_cache.code_identities(raw_code_paths)
    if preprocess_cache is not None:
        # This must precede every source path operation.  In particular, a
        # sealed-test path is never resolved, stat'ed, hashed, or probed.
        artifact_cache.require_working_split(split)
        artifact_cache.require_disjoint_roots(
            preprocess_cache, output, video
        )
    if sample_fps <= 0.0:
        raise RuntimeError("sample_fps must be positive")
    if global_policy_weight <= 0.0:
        raise RuntimeError("global_policy_weight must be positive")
    if min_context_frames < 2:
        raise RuntimeError("min_context_frames must be at least 2")
    if not 0 <= png_compression <= 9:
        raise RuntimeError("png_compression must be between 0 and 9")
    if write_workers < 1:
        raise RuntimeError("write_workers must be at least 1")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    video = Path(video).resolve(strict=True)
    source_snapshot = artifact_cache.source_file_snapshot(video)
    color_probe = video_color.probe_sdr_input(video, ffprobe, input_color)
    video_size = source_snapshot["bytes"]
    video_sha256 = source_snapshot["sha256"]
    cache = None
    raw_identity = None
    raw_root = None
    raw_manifest = None
    raw_receipt = None
    cache_summary = {
        "enabled": False,
        "contract": raw_movie.CONTRACT,
        "key_sha256": None,
        "status": "disabled",
        "payload_bytes": 0,
        "file_count": 0,
    }
    if preprocess_cache is not None:
        cache = artifact_cache.DirectoryArtifactCache(preprocess_cache)
        raw_runtime_identity = raw_movie.runtime_identity(fresh=True)
        raw_identity = raw_movie.identity(
            video_size=video_size, video_sha256=video_sha256, split=split,
            start_seconds=start_seconds,
            end_margin_seconds=end_margin_seconds,
            output_width=output_width, png_compression=png_compression,
            color_probe=color_probe, code_identity=raw_code_identity,
            runtime_identity_value=raw_runtime_identity,
        )
        if output.exists():
            # The legacy contract permits an existing empty destination.
            output.rmdir()
        raw_root, raw_manifest, cache_summary, raw_receipt = raw_movie.resolve(
            cache=cache, identity_value=raw_identity, video=video,
            output_parent=output.parent, output_width=output_width,
            start_seconds=start_seconds,
            end_margin_seconds=end_margin_seconds,
            png_compression=png_compression, write_workers=write_workers,
            code_identity=raw_code_identity,
            source_snapshot=source_snapshot,
            runtime_identity_value=raw_runtime_identity,
        )
        if cache_summary["status"] == "hit":
            artifact_cache.verify_source_file_snapshot(
                video, source_snapshot
            )
    if cache_observer is not None:
        cache_observer(dict(cache_summary))
    output.mkdir(parents=True, exist_ok=True)

    capture = None
    if raw_manifest is None:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            capture.release()
            shutil.rmtree(output)
            raise RuntimeError(f"cannot open monocular movie: {video}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_frames_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if source_fps <= 0.0:
            capture.release()
            shutil.rmtree(output)
            raise RuntimeError("movie has an invalid frame rate")
    else:
        source_fps = float(raw_manifest["source_fps"])
        source_frames_reported = raw_manifest["source_frames_reported"]
    stride = max(1, round(source_fps / sample_fps))
    if raw_manifest is None:
        start_frame = max(0, round(start_seconds * source_fps))
        end_frame = (source_frames_reported - max(
            0, round(end_margin_seconds * source_fps)
        ) if source_frames_reported > 0 else None)
        if end_frame is not None and start_frame >= end_frame:
            capture.release()
            shutil.rmtree(output)
            raise RuntimeError("movie trim removes all frames")
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = raw_manifest["start_frame"]
        end_frame = start_frame + raw_manifest["decoded_frame_count"]
    writer = _BoundedPngWriter(write_workers)
    authenticated_raw_files = ({
        row["path"]: row for row in raw_receipt["files"]
    } if raw_receipt is not None else {})

    production = production_id or domain
    rows = []
    dropped = []
    previous_gray = None
    shot = None
    shot_index = -1
    frame_index = start_frame

    def begin_shot(source_index, transition_score, frame=None, raw_row=None):
        nonlocal shot_index
        shot_index += 1
        clip_name = f"{domain}_shot_{shot_index:04d}"
        partial = output / f".{clip_name}.partial"
        _assert_direct_child(partial, output)
        if partial.exists():
            shutil.rmtree(partial)
        _assert_direct_child(partial, output)
        partial.mkdir()
        if raw_row is None:
            prepared = resized_frame(frame, output_width)
            prepared_width = int(prepared.shape[1])
            prepared_height = int(prepared.shape[0])
        else:
            prepared_width = raw_row["width"]
            prepared_height = raw_row["height"]
        return {
            "shot": shot_index,
            "clip": clip_name,
            "partial": partial,
            "source_start_frame": source_index,
            "source_end_frame": source_index,
            "max_change_score": transition_score,
            "context_frames": 0,
            "label_frame_ids": [],
            "width": prepared_width,
            "height": prepared_height,
        }

    def append_frame(state, source_index, frame=None, raw_row=None):
        local_id = state["context_frames"]
        destination = state["partial"] / f"frame_{local_id:05d}.png"
        if raw_row is None:
            prepared = resized_frame(frame, output_width)
            if (prepared.shape[1] != state["width"] or
                    prepared.shape[0] != state["height"]):
                raise RuntimeError("movie resolution changed inside a detected shot")
            writer.submit(
                destination, prepared,
                [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
            )
        else:
            if (raw_row["width"] != state["width"] or
                    raw_row["height"] != state["height"]):
                raise RuntimeError("movie resolution changed inside a detected shot")
            shutil.copy2(raw_root / raw_row["path"], destination)
            expected = authenticated_raw_files.get(raw_row["path"])
            if (expected is None or
                    destination.stat().st_size != expected["bytes"] or
                    sha256(destination) != expected["sha256"]):
                raise RuntimeError(
                    "raw movie cache frame changed during materialization"
                )
        if local_id == 0 or local_id % stride == 0:
            state["label_frame_ids"].append(local_id)
        state["context_frames"] += 1
        state["source_end_frame"] = source_index

    def finish_shot(state):
        if state is None:
            return
        # A clip only becomes visible under its final name after every queued
        # frame has been encoded successfully.
        writer.wait()
        if state["context_frames"] < min_context_frames:
            dropped.append({
                "shot": state["shot"],
                "source_start_frame": state["source_start_frame"],
                "source_end_frame": state["source_end_frame"],
                "context_frames": state["context_frames"],
                "reason": "insufficient-temporal-context",
            })
            _assert_direct_child(state["partial"], output)
            shutil.rmtree(state["partial"])
            return
        clip = output / state["clip"]
        _assert_direct_child(clip, output)
        if clip.exists():
            raise RuntimeError(f"refusing to replace prepared clip: {clip}")
        # Keep this file deliberately minimal: its exact bytes are authenticated by the harness.
        # Sampling/provenance details belong in meta.json and the dataset manifest.
        label_contract = {
            "schema": 1,
            "frame_ids": state["label_frame_ids"],
        }
        meta = {
            "schema": 2,
            "name": f"{name}-shot-{state['shot']:04d}",
            "dataset": name,
            "domain": domain,
            "production_id": production,
            "film_id": production,
            "homepage": homepage or None,
            "license": license_name,
            "purpose": "artistic-policy monocular render-feasibility supervision",
            "policy_role": policy_role,
            "global_policy_weight": global_policy_weight,
            "source_kind": "mono-video",
            "temporal_contract": "full-cadence-shot",
            "shot": state["shot"],
            "source_start_frame": state["source_start_frame"],
            "source_end_frame": state["source_end_frame"],
            "context_fps": source_fps,
            "label_sampling": "shot-first-frame-plus-shot-local-cadence",
            "label_stride": stride,
            "requested_label_fps": sample_fps,
            "context_frame_count": state["context_frames"],
            "label_frame_count": len(state["label_frame_ids"]),
            "source_width": state["width"],
            "source_height": state["height"],
            "split": split,
            "required_gt_stereo": False,
            "required_temporal_evidence": True,
            "color_contract": color_probe["dataset_color_contract"],
            "source_color_probe": color_probe,
            "write_worker_count": write_workers,
        }
        write_json(state["partial"] / "label_frames.json", label_contract)
        write_json(state["partial"] / "meta.json", meta)
        _assert_direct_child(state["partial"], output)
        _assert_direct_child(clip, output)
        state["partial"].rename(clip)
        rows.append({
            "clip": state["clip"],
            "shot": state["shot"],
            "source_start_frame": state["source_start_frame"],
            "source_end_frame": state["source_end_frame"],
            "context_frames": state["context_frames"],
            "label_frames": len(state["label_frame_ids"]),
            "split": split,
        })

    try:
        if raw_manifest is not None:
            for raw_row in raw_manifest["frames"]:
                score = float(raw_row["change_score"])
                source_index = raw_row["source_frame"]
                if raw_row["ordinal"] == 0 or score >= cut_threshold:
                    finish_shot(shot)
                    shot = begin_shot(
                        source_index, score, raw_row=raw_row
                    )
                else:
                    shot["max_change_score"] = max(
                        shot["max_change_score"], score
                    )
                append_frame(
                    shot, source_index, raw_row=raw_row
                )
                frame_index = source_index + 1
        else:
            while end_frame is None or frame_index < end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                current_gray = analysis_gray(frame)
                score = (0.0 if previous_gray is None
                         else change_score(previous_gray, current_gray))
                if previous_gray is None or score >= cut_threshold:
                    finish_shot(shot)
                    shot = begin_shot(frame_index, score, frame)
                else:
                    shot["max_change_score"] = max(
                        shot["max_change_score"], score
                    )
                append_frame(shot, frame_index, frame)
                previous_gray = current_gray
                frame_index += 1
        finish_shot(shot)
        writer.close()
        if raw_manifest is None:
            artifact_cache.verify_source_file_snapshot(
                video, source_snapshot
            )
    except BaseException:
        writer.abort()
        if output.exists():
            shutil.rmtree(output)
        raise
    finally:
        if capture is not None:
            capture.release()

    if not rows:
        shutil.rmtree(output)
        raise RuntimeError("movie preparation produced no temporally valid shots")
    decoded_frame_count = frame_index - start_frame
    manifest = {
        "schema": 2,
        "dataset": name,
        "domain": domain,
        "production_id": production,
        "source_kind": "mono-video",
        "temporal_contract": "full-cadence-shot",
        "policy_role": policy_role,
        "homepage": homepage or None,
        "license": license_name,
        "split": split,
        "video": str(video.resolve()),
        "video_size": video_size,
        "video_sha256": video_sha256,
        "source_fps": source_fps,
        "source_frames_reported": source_frames_reported,
        "decoded_frame_count": decoded_frame_count,
        "start_seconds": start_seconds,
        "end_margin_seconds": end_margin_seconds,
        "label_stride": stride,
        "requested_label_fps": sample_fps,
        "label_sampling": "shot-first-frame-plus-shot-local-cadence",
        "cut_threshold": cut_threshold,
        "output_width": output_width,
        "png_compression": png_compression,
        "min_context_frames": min_context_frames,
        "global_policy_weight": global_policy_weight,
        "color_contract": color_probe["dataset_color_contract"],
        "source_color_probe": color_probe,
        "write_worker_count": write_workers,
        "sequences": rows,
        "dropped_shots": dropped,
        "shot_count": len(rows),
        "dropped_shot_count": len(dropped),
        "context_frame_count": sum(row["context_frames"] for row in rows),
        "label_frame_count": sum(row["label_frames"] for row in rows),
        "split_rule": "the complete production has one split; never split its shots",
    }
    try:
        write_json(output / "dataset_manifest.json", manifest)
        if cache is not None:
            artifact_cache.verify_code_identities(
                code_paths, code_identity
            )
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--cut-threshold", type=float, default=0.18)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-margin-seconds", type=float, default=0.0)
    parser.add_argument("--split", choices=("training", "development", "test"),
                        default="training")
    parser.add_argument("--production-id", default="")
    parser.add_argument("--homepage", default="")
    parser.add_argument("--license", default="user-provided; verify rights")
    parser.add_argument("--policy-role", default="cinematic_training")
    parser.add_argument("--global-policy-weight", type=float, default=1.0)
    parser.add_argument("--min-context-frames", type=int, default=3)
    parser.add_argument("--png-compression", type=int, default=1)
    parser.add_argument("--write-workers", type=int, default=4)
    parser.add_argument(
        "--preprocess-cache", type=Path,
        help=(
            "optional authenticated content-addressed cache; available only "
            "for training/development sources"
        ),
    )
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument(
        "--input-color", choices=("auto", "sdr"), default="auto",
        help="auto rejects HDR/high-bit-depth input; sdr records a reviewed SDR override",
    )
    args = parser.parse_args()
    cache_events = []
    manifest = prepare(
        args.video, args.output, args.name, args.domain, args.sample_fps,
        args.cut_threshold, args.output_width, args.start_seconds,
        args.end_margin_seconds, args.split, args.production_id, args.homepage,
        args.license, args.policy_role, args.global_policy_weight,
        args.min_context_frames, args.png_compression, args.ffprobe,
        args.input_color, args.write_workers, args.preprocess_cache,
        cache_events.append,
    )
    print(json.dumps({
        "preprocessing_cache": cache_events[-1],
        "dataset_manifest": manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
