#!/usr/bin/env python3
"""Prepare left/right training clips from Apple MV-HEVC spatial videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


SVD_HOME = "https://cd-athena.github.io/SVD/"
SVD_DOWNLOAD = "https://ftp.itec.aau.at/datasets/SVD/"


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(map(str, command))}\n"
            f"{(process.stdout + process.stderr)[-4000:]}"
        )
    return process.stdout


def probe_video(ffprobe: Path, source: Path):
    payload = json.loads(run([
        str(ffprobe), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(source),
    ]))
    videos = [
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(videos) != 1:
        raise RuntimeError(f"{source}: expected one MV-HEVC video stream")
    stream = videos[0]
    view_positions = str(stream.get("view_pos_available", ""))
    view_ids = str(stream.get("view_ids_available", ""))
    positions = {item.strip() for item in view_positions.split(",")}
    if not view_ids or "," not in view_ids or positions != {"1", "2"}:
        raise RuntimeError(
            f"{source}: missing left/right MV-HEVC views "
            f"(ids={view_ids!r}, positions={view_positions!r})"
        )
    return payload, stream


def extract_view(ffmpeg: Path, source: Path, position: str, output: Path,
                 sample_fps: float, output_width: int):
    output.mkdir(parents=True, exist_ok=True)
    scale = f"scale={output_width}:-2:flags=lanczos" if output_width else "null"
    run([
        str(ffmpeg), "-v", "error", "-y", "-i", str(source),
        "-map", f"0:v:0:vpos:{position}",
        "-vf", f"fps={sample_fps},{scale}", "-fps_mode", "passthrough",
        str(output / "frame_%05d.png"),
    ])


def group_name(path: Path, source_root: Path):
    relative = path.relative_to(source_root)
    return relative.parts[0].lower() if len(relative.parts) > 1 else "spatial"


def prepared_name(source: Path, source_root: Path):
    group = group_name(source, source_root)
    return f"svd_{group}_{source.stem.lower()}"


def prepare(source_root: Path, output: Path, ffmpeg: Path, ffprobe: Path,
            sample_fps=2.0, output_width=1280, validation_modulus=5,
            groups=(), limit_per_group=0):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output must be empty: {output}")
    for path, description in ((ffmpeg, "ffmpeg"), (ffprobe, "ffprobe")):
        if not path.is_file():
            raise RuntimeError(f"missing {description}: {path}")
    output.mkdir(parents=True, exist_ok=True)
    selected_groups = {item.lower() for item in groups}
    sources = sorted(source_root.rglob("*.mov"))
    if selected_groups:
        sources = [
            source for source in sources
            if group_name(source, source_root) in selected_groups
        ]
    if limit_per_group:
        retained = []
        counts = {}
        for source in sources:
            group = group_name(source, source_root)
            counts[group] = counts.get(group, 0) + 1
            if counts[group] <= limit_per_group:
                retained.append(source)
        sources = retained
    if not sources:
        raise RuntimeError(f"no MV-HEVC .mov files found under {source_root}")

    sequences = []
    for index, source in enumerate(sources, 1):
        group = group_name(source, source_root)
        clip_name = prepared_name(source, source_root)
        clip = output / clip_name
        _, stream = probe_video(ffprobe, source)
        extract_view(
            ffmpeg, source, "left", clip, sample_fps, output_width
        )
        extract_view(
            ffmpeg, source, "right", clip / "gt_right",
            sample_fps, output_width,
        )
        left = sorted(clip.glob("frame_*.png"))
        right = sorted((clip / "gt_right").glob("frame_*.png"))
        if not left or [item.name for item in left] != [item.name for item in right]:
            raise RuntimeError(f"{source}: extracted left/right frames do not match")
        numeric_id = int(source.stem) if source.stem.isdigit() else index
        split = (
            "validation" if numeric_id % validation_modulus == 0 else "training"
        )
        source_url = (
            f"{SVD_DOWNLOAD}{source.relative_to(source_root).as_posix()}"
        )
        side_data = stream.get("side_data_list", [])
        stereo_metadata = next(
            (item for item in side_data
             if item.get("side_data_type") == "Stereo 3D"), {}
        )
        spherical_metadata = next(
            (item for item in side_data
             if item.get("side_data_type") == "Spherical Mapping"), {}
        )
        meta = {
            "schema": 1,
            "name": clip_name,
            "dataset": "SVD Spatial Video Dataset",
            "domain": f"svd_{group}",
            "film_id": clip_name,
            "homepage": SVD_HOME,
            "source_url": source_url,
            "source_sha256": sha256(source),
            "license": "CC BY-NC 4.0",
            "purpose": "artistic-policy spatial-video local geometry",
            "policy_role": "vr_parallel",
            "global_policy_weight": 0.0,
            "projection": spherical_metadata.get("projection", "rectilinear"),
            "source_format": "MV-HEVC",
            "view_positions": stream.get("view_pos_available"),
            "primary_eye": stereo_metadata.get("primary_eye"),
            "baseline_micrometers": stereo_metadata.get("baseline"),
            "horizontal_field_of_view": stereo_metadata.get(
                "horizontal_field_of_view"
            ),
            "source_width": stream.get("width"),
            "source_height": stream.get("height"),
            "pixel_format": stream.get("pix_fmt"),
            "color_range": stream.get("color_range"),
            "color_space": stream.get("color_space"),
            "color_transfer": stream.get("color_transfer"),
            "color_primaries": stream.get("color_primaries"),
            "sample_fps": sample_fps,
            "frame_count": len(left),
            "split": split,
            "required_gt_stereo": True,
        }
        (clip / "meta.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        sequences.append({
            "clip": clip_name,
            "source": str(source.resolve()),
            "source_url": source_url,
            "source_sha256": meta["source_sha256"],
            "group": group,
            "frames": len(left),
            "split": split,
        })
        print(f"[{index}/{len(sources)}] {clip_name}: {len(left)} pairs", flush=True)

    manifest = {
        "schema": 1,
        "dataset": "SVD Spatial Video Dataset",
        "homepage": SVD_HOME,
        "license": "CC BY-NC 4.0",
        "policy_role": "vr_parallel",
        "global_policy_weight": 0.0,
        "source_root": str(source_root.resolve()),
        "ffmpeg": str(ffmpeg.resolve()),
        "ffmpeg_sha256": sha256(ffmpeg),
        "ffprobe": str(ffprobe.resolve()),
        "ffprobe_sha256": sha256(ffprobe),
        "sample_fps": sample_fps,
        "output_width": output_width,
        "validation_modulus": validation_modulus,
        "sequences": sequences,
        "clip_count": len(sequences),
        "frame_count": sum(item["frames"] for item in sequences),
        "training_frames": sum(
            item["frames"] for item in sequences if item["split"] == "training"
        ),
        "validation_frames": sum(
            item["frames"] for item in sequences
            if item["split"] == "validation"
        ),
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--ffprobe", required=True, type=Path)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--output-width", type=int, default=1280)
    parser.add_argument("--validation-modulus", type=int, default=5)
    parser.add_argument("--groups", default="")
    parser.add_argument("--limit-per-group", type=int, default=0)
    args = parser.parse_args()
    manifest = prepare(
        args.source, args.output, args.ffmpeg, args.ffprobe,
        args.sample_fps, args.output_width, args.validation_modulus,
        tuple(item.strip() for item in args.groups.split(",") if item.strip()),
        args.limit_per_group,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
