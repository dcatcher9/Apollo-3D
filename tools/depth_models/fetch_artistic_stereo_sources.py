#!/usr/bin/env python3
"""Fetch and audit licensed authored-stereo sources outside the repository."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np


SPLITS = {"training", "development", "test"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command):
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        output = (process.stdout + process.stderr)[-8000:]
        raise RuntimeError(
            f"command failed ({process.returncode}): "
            f"{' '.join(map(str, command))}\n{output}"
        )
    return process.stdout


def load_catalog(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise RuntimeError(f"unsupported source catalog schema: {path}")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("source catalog has no sources")
    identifiers = set()
    production_splits = {}
    production_sources = {}
    sealed = set()
    sealed_groups = set()
    for source in sources:
        identifier = source.get("id")
        if not identifier or identifier in identifiers:
            raise RuntimeError(f"missing or duplicate source id: {identifier!r}")
        identifiers.add(identifier)
        admission = source.get("admission", "")
        split = source.get("split")
        if admission == "global_policy":
            if split not in SPLITS:
                raise RuntimeError(f"{identifier}: invalid global-policy split")
            if not source.get("complete_production"):
                raise RuntimeError(
                    f"{identifier}: global-policy source is not a complete production"
                )
            if float(source.get("global_policy_weight", 0.0)) <= 0.0:
                raise RuntimeError(f"{identifier}: global-policy weight is not positive")
            if not source.get("source_group"):
                raise RuntimeError(f"{identifier}: missing source group")
            if not source.get("license") or not source.get("license_url"):
                raise RuntimeError(f"{identifier}: missing license provenance")
            if source.get("eye_order") not in {"first-left", "first-right"}:
                raise RuntimeError(f"{identifier}: eye order is not verified")
            if float(source.get("eye_display_aspect_ratio", 0.0)) <= 0.0:
                raise RuntimeError(f"{identifier}: display eye aspect is missing")
            if split == "test":
                sealed.add(source.get("production_id"))
                sealed_groups.add(source["source_group"])
        production = source.get("production_id")
        if production and split:
            previous = production_splits.setdefault(production, split)
            if previous != split:
                raise RuntimeError(
                    f"production {production!r} leaks across {previous} and {split}"
                )
            owner = production_sources.setdefault(production, identifier)
            if owner != identifier:
                raise RuntimeError(
                    f"production {production!r} has duplicate sources "
                    f"{owner!r} and {identifier!r}"
                )
    if len(sealed) < 2:
        raise RuntimeError("source catalog needs two sealed authored test productions")
    if len(sealed_groups) < 2:
        raise RuntimeError("sealed tests need two independent source groups")
    return payload


def request(url, offset=0):
    headers = {"User-Agent": "Apollo-3D-artistic-data/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def download_http(urls, destination: Path):
    partial = destination.with_suffix(destination.suffix + ".part")
    marker = partial.with_suffix(partial.suffix + ".url")
    errors = []
    for url in urls:
        previous_url = (
            marker.read_text(encoding="utf-8").strip()
            if marker.is_file() else ""
        )
        if partial.exists() and previous_url != url:
            partial.unlink()
        marker.write_text(url + "\n", encoding="utf-8")
        offset = partial.stat().st_size if partial.exists() else 0
        try:
            with urllib.request.urlopen(request(url, offset), timeout=90) as response:
                resumed = offset and response.status == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    offset = 0
                transferred = offset
                next_report = transferred + 64 * 1024 * 1024
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        transferred += len(chunk)
                        if transferred >= next_report:
                            print(
                                f"  downloaded {transferred / (1024 ** 2):.0f} MiB",
                                flush=True,
                            )
                            next_report += 64 * 1024 * 1024
            partial.replace(destination)
            marker.unlink(missing_ok=True)
            return url
        except (OSError, urllib.error.URLError) as error:
            errors.append(f"{url}: {error}")
            print(f"  source failed, trying fallback: {error}", flush=True)
    raise RuntimeError("all download URLs failed:\n" + "\n".join(errors))


def download_ytdlp(source, directory: Path, ffmpeg: Path | None):
    retrieval = source["retrieval"]
    stem = retrieval["filename_stem"]
    output = directory / f"{stem}.%(ext)s"
    command = [
        sys.executable, "-m", "yt_dlp", "--no-playlist",
        "--write-info-json", "--no-overwrites",
        "--merge-output-format", "mkv",
        "-f", "bestvideo*+bestaudio/best", "-o", str(output),
    ]
    if ffmpeg:
        command += ["--ffmpeg-location", str(ffmpeg.parent)]
    command.append(retrieval["url"])
    run(command)
    videos = sorted(
        item for item in directory.glob(f"{stem}.*")
        if item.suffix.lower() in VIDEO_SUFFIXES
    )
    if len(videos) != 1:
        raise RuntimeError(
            f"{source['id']}: expected one downloaded video, found {videos}"
        )
    return videos[0], retrieval["url"]


def combine_http_pair(source, directory: Path, ffmpeg: Path, ffprobe: Path):
    retrieval = source["retrieval"]
    paths = []
    used_urls = []
    for key in ("first", "second"):
        item = retrieval[key]
        path = directory / item["filename"]
        used_url = "existing"
        if not path.is_file():
            used_url = download_http(item["urls"], path)
        paths.append(path)
        used_urls.append(used_url)
    probes = [probe(ffprobe, path) for path in paths]
    dimensions = [
        (int(stream["width"]), int(stream["height"]))
        for _, stream in probes
    ]
    if dimensions[0] != dimensions[1]:
        raise RuntimeError(
            f"{source['id']}: separate-eye dimensions differ: {dimensions}"
        )
    durations = [
        float(payload.get("format", {}).get("duration", 0.0))
        for payload, _ in probes
    ]
    if min(durations) <= 0.0 or abs(durations[0] - durations[1]) > 0.05:
        raise RuntimeError(
            f"{source['id']}: separate-eye durations differ: {durations}"
        )
    video = directory / retrieval["filename"]
    identities = [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    identity_path = video.with_suffix(video.suffix + ".inputs.json")
    previous = None
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
    if not video.is_file() or previous != identities:
        temporary = video.with_suffix(".building" + video.suffix)
        temporary.unlink(missing_ok=True)
        run([
            str(ffmpeg), "-nostdin", "-loglevel", "error", "-y",
            "-i", str(paths[0]), "-i", str(paths[1]),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
            "-map", "[v]", "-map", "0:a?", "-c:v", "libx264",
            "-preset", "medium", "-crf", "10", "-pix_fmt", "yuv420p",
            "-c:a", "copy", str(temporary),
        ])
        temporary.replace(video)
        identity_path.write_text(
            json.dumps(identities, indent=2) + "\n", encoding="utf-8"
        )
    return video, used_urls, identities


def probe(ffprobe: Path, video: Path):
    payload = json.loads(run([
        str(ffprobe), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(video),
    ]))
    streams = [
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    ]
    if len(streams) != 1:
        raise RuntimeError(f"{video}: expected exactly one video stream")
    return payload, streams[0]


def split_eyes(frame, layout):
    height, width = frame.shape[:2]
    if layout == "side-by-side":
        if width % 2:
            raise RuntimeError(f"SBS source has odd width: {width}")
        return frame[:, :width // 2], frame[:, width // 2:]
    if layout == "above-below":
        if height % 2:
            raise RuntimeError(f"above/below source has odd height: {height}")
        return frame[:height // 2], frame[height // 2:]
    raise RuntimeError(f"unsupported stereo layout: {layout}")


def feature_deltas(first, second):
    scale = min(1.0, 960.0 / first.shape[1])
    size = (round(first.shape[1] * scale), round(first.shape[0] * scale))
    first_gray = cv2.cvtColor(cv2.resize(first, size), cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(cv2.resize(second, size), cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT_create(nfeatures=2500)
    first_keys, first_desc = detector.detectAndCompute(first_gray, None)
    second_keys, second_desc = detector.detectAndCompute(second_gray, None)
    if first_desc is None or second_desc is None:
        return []
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        first_desc, second_desc, k=2
    )
    deltas = []
    for pair in matches:
        if len(pair) != 2 or pair[0].distance >= 0.72 * pair[1].distance:
            continue
        match = pair[0]
        x1, y1 = first_keys[match.queryIdx].pt
        x2, y2 = second_keys[match.trainIdx].pt
        deltas.append(((x2 - x1) / size[0], (y2 - y1) / size[1]))
    return deltas


def audit_video(source, video: Path, ffprobe: Path, directory: Path):
    payload, stream = probe(ffprobe, video)
    width = int(stream["width"])
    height = int(stream["height"])
    split_eyes(np.zeros((height, width, 3), np.uint8), source["layout"])
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode downloaded video: {video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        raise RuntimeError(f"downloaded video reports no frames: {video}")
    indices = sorted({round((frame_count - 1) * position) for position in (
        0.05, 0.25, 0.5, 0.75, 0.95
    )})
    all_deltas = []
    frame_vertical_medians = []
    difference = []
    contact_rows = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"cannot decode frame {index}: {video}")
        first, second = split_eyes(frame, source["layout"])
        first_small = cv2.resize(first, (480, round(480 * first.shape[0] / first.shape[1])))
        second_small = cv2.resize(second, first_small.shape[1::-1])
        first_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY)
        second_gray = cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY)
        difference.append(float(np.mean(cv2.absdiff(first_gray, second_gray))) / 255.0)
        frame_deltas = feature_deltas(first, second)
        all_deltas.extend(frame_deltas)
        if frame_deltas:
            frame_vertical_medians.append(float(np.median(np.abs(
                np.asarray(frame_deltas, np.float32)[:, 1]
            ))) * 100.0)
        anaglyph = np.empty_like(first_small)
        anaglyph[..., 2] = first_gray
        anaglyph[..., 1] = second_gray
        anaglyph[..., 0] = second_gray
        row = np.concatenate((first_small, second_small, anaglyph), axis=1)
        cv2.putText(
            row, f"frame {index}: first | second | first-red anaglyph",
            (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        contact_rows.append(row)
    capture.release()
    if max(difference, default=0.0) < 0.002:
        raise RuntimeError(f"{video}: the two stored eyes appear identical")
    deltas = np.asarray(all_deltas, np.float32)
    if deltas.shape[0] < 30:
        raise RuntimeError(
            f"{video}: too few cross-eye feature matches ({deltas.shape[0]})"
        )
    horizontal = deltas[:, 0]
    vertical = np.abs(deltas[:, 1])
    sheet = np.concatenate(contact_rows, axis=0)
    contact_path = directory / "stereo_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    stored_eye_width = width // 2 if source["layout"] == "side-by-side" else width
    stored_eye_height = height if source["layout"] == "side-by-side" else height // 2
    audit = {
        "video": str(video.resolve()),
        "video_sha256": sha256(video),
        "video_size": video.stat().st_size,
        "container": payload.get("format", {}).get("format_name"),
        "duration_seconds": float(payload.get("format", {}).get("duration", 0.0)),
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "color_range": stream.get("color_range"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "stored_width": width,
        "stored_height": height,
        "stored_eye_width": stored_eye_width,
        "stored_eye_height": stored_eye_height,
        "stored_eye_aspect_ratio": stored_eye_width / stored_eye_height,
        "display_eye_aspect_ratio": source.get("eye_display_aspect_ratio", 0.0),
        "frame_count": frame_count,
        "sampled_frames": indices,
        "mean_cross_eye_difference": float(np.mean(difference)),
        "feature_match_count": int(deltas.shape[0]),
        "second_minus_first_horizontal_pct": {
            "p05": float(np.percentile(horizontal, 5) * 100.0),
            "p50": float(np.percentile(horizontal, 50) * 100.0),
            "p95": float(np.percentile(horizontal, 95) * 100.0),
        },
        "absolute_vertical_mismatch_pct": {
            "p50": float(np.percentile(vertical, 50) * 100.0),
            "p95": float(np.percentile(vertical, 95) * 100.0),
        },
        "sample_vertical_median_pct": {
            "p50": float(np.percentile(frame_vertical_medians, 50)),
            "p95": float(np.percentile(frame_vertical_medians, 95)),
        },
        "declared_eye_order": source.get("eye_order"),
        "source_layout": source.get("source_layout", source["layout"]),
        "contact_sheet": str(contact_path.resolve()),
    }
    (directory / "video_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def validate_audit(source, audit):
    if source.get("admission") != "global_policy":
        return
    vertical = audit["sample_vertical_median_pct"]
    if vertical["p50"] > 1.0 or vertical["p95"] > 1.0:
        raise RuntimeError(
            f"{source['id']}: stereo rectification failed: vertical mismatch "
            f"p50={vertical['p50']:.3f}%, p95={vertical['p95']:.3f}%"
        )


def fetch_source(source, output: Path, ffmpeg: Path | None, ffprobe: Path):
    directory = output / source["id"]
    directory.mkdir(parents=True, exist_ok=True)
    retrieval = source["retrieval"]
    pair_inputs = None
    if retrieval["kind"] == "http":
        video = directory / retrieval["filename"]
        used_url = None
        if not video.is_file():
            used_url = download_http(retrieval["urls"], video)
        else:
            used_url = "existing"
    elif retrieval["kind"] == "yt-dlp":
        video, used_url = download_ytdlp(source, directory, ffmpeg)
    elif retrieval["kind"] == "http-pair":
        if ffmpeg is None:
            raise RuntimeError(f"{source['id']}: --ffmpeg is required for http-pair")
        video, used_url, pair_inputs = combine_http_pair(
            source, directory, ffmpeg, ffprobe
        )
    else:
        raise RuntimeError(
            f"{source['id']}: retrieval kind {retrieval['kind']!r} is not automatic"
        )
    audit_source = dict(source)
    if source["layout"] == "separate-files":
        audit_source["layout"] = "side-by-side"
        audit_source["source_layout"] = source["layout"]
    audit = audit_video(audit_source, video, ffprobe, directory)
    validate_audit(source, audit)
    record = {
        "schema": 1,
        "source": source,
        "retrieved_from": used_url,
        "retrieved_at_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "audit": audit,
    }
    if pair_inputs is not None:
        record["pair_inputs"] = pair_inputs
    (directory / "source_record.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main():
    default_catalog = Path(__file__).with_name("artistic_stereo_sources.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=default_catalog)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--all-auto", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", required=True, type=Path)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    by_id = {source["id"]: source for source in catalog["sources"]}
    if args.list:
        for source in catalog["sources"]:
            print(
                f"{source['id']}: {source['retrieval']['kind']}; "
                f"{source.get('split')}; {source['admission']}"
            )
        return
    selected = list(args.source)
    if args.all_auto:
        selected.extend(
            source["id"] for source in catalog["sources"]
            if source["retrieval"]["kind"] in {"http", "http-pair", "yt-dlp"}
            and not source.get("admission", "").startswith("excluded_")
        )
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise RuntimeError("select --source ID or --all-auto")
    missing = sorted(set(selected) - set(by_id))
    if missing:
        raise RuntimeError(f"unknown source ids: {', '.join(missing)}")
    if not args.ffprobe.is_file():
        raise RuntimeError(f"missing ffprobe: {args.ffprobe}")
    if any(by_id[item]["retrieval"]["kind"] == "yt-dlp" for item in selected):
        try:
            __import__("yt_dlp")
        except ImportError as error:
            raise RuntimeError(
                "yt-dlp is required in the active Python environment"
            ) from error
        if args.ffmpeg and not args.ffmpeg.is_file():
            raise RuntimeError(f"missing ffmpeg: {args.ffmpeg}")
    if any(by_id[item]["retrieval"]["kind"] == "http-pair" for item in selected):
        if not args.ffmpeg or not args.ffmpeg.is_file():
            raise RuntimeError("--ffmpeg is required for separate-eye sources")
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, identifier in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {identifier}", flush=True)
        records.append(fetch_source(
            by_id[identifier], args.output, args.ffmpeg, args.ffprobe
        ))
    summary = {
        "schema": 1,
        "catalog": str(args.catalog.resolve()),
        "catalog_sha256": sha256(args.catalog),
        "sources": [record["source"]["id"] for record in records],
        "records": [
            str((args.output / item["source"]["id"] / "source_record.json").resolve())
            for item in records
        ],
    }
    (args.output / "fetch_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
