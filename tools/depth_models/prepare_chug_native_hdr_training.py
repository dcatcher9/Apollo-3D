#!/usr/bin/env python3
"""Prepare a bounded native-PQ CHUG bootstrap as FP16 Windows scRGB.

This tool never turns HDR into an SDR model source.  Each output clip contains
two distinct artifacts:

* ``frame_*.png``: the perceptual sRGB preview used by image-relative metrics;
* ``model_source/frame_*.scrgb16``: tightly packed RGBA16F linear scRGB used by
  the production depth path and render harness.

FFmpeg is pinned to decode HEVC and reconstruct limited-range BT.2020 NCL into
nonlinear float RGB.  ST-2084, the linear BT.2020-to-Rec.709 transform, 80-nit
scRGB normalization, half-float storage, and the evaluator preview are applied
explicitly here and are bound into every derivative manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import numpy as np
from PIL import Image

import depth_input_color as input_color
import native_hdr_capture

SBSBENCH_DIR = Path(__file__).resolve().parents[1] / "sbsbench"
sys.path.insert(0, str(SBSBENCH_DIR))
import sbsbench  # noqa: E402
import run_eval  # noqa: E402


PREPARATION_SCHEMA = 3
CONVERSION_SCHEMA = 1
PREPARATION_CONTRACT = "apollo-chug-native-pq-training-v3"
# Keep the already-audited clip choice stable when the storage layout changes.
SELECTION_SEED_CONTRACT = "apollo-chug-native-pq-training-v1"
CONVERSION_CONTRACT = "pq-bt2020nc-to-windows-scrgb16-v1"
FRAME_SELECTION_CONTRACT = "label-centered-source-frame-windows-v2"
BOOTSTRAP_MANIFEST = "native_hdr_bootstrap_manifest.json"
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TRAINING_COUNT = 12
DEVELOPMENT_COUNT = 4
LABELS_PER_CLIP = 5
TEMPORAL_WINDOW_RADIUS = 1
FLOW_SUPPORT_SEARCH_RADIUS_FRAMES = 12
FLOW_SUPPORT_SELECTION_CONTRACT = "nearest-valid-source-flow-window-v1"
FLOW_SUPPORT_CONTRACT = "sbsbench-flow-temporal-source-reliability-v1"
FLOW_TEMPORAL_MIN_SUPPORT = 0.1
SCRGB_BYTES_PER_PIXEL = 8
SPLITS = ("training", "development")

BT2020_TO_REC709 = np.asarray(
    input_color.load_color_contract()["native_pq_in_windows_hdr"]
    ["linear_bt2020_to_linear_rec709_d65"],
    dtype=np.float32,
)
PQ = input_color.load_color_contract()["native_pq_in_windows_hdr"]["pq_eotf"]
REC709_LUMA = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flow_support_metric_sha256() -> str:
    """Bind curation to the complete automatic metric/gating contract."""
    return run_eval.metric_contract_sha()


def stable_hash(*parts) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def read_json(path: Path, description: str):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {description}: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is not an object: {path}")
    return payload


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{uuid.uuid4().hex}")
    try:
        temporary.write_bytes(native_hdr_capture.canonical_json_bytes(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def publish_directory(staging: Path, destination: Path) -> None:
    backup = destination.with_name(
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    moved = False
    try:
        if destination.exists() or destination.is_symlink():
            destination.replace(backup)
            moved = True
        staging.replace(destination)
    except BaseException:
        if moved and not destination.exists():
            backup.replace(destination)
        raise
    if moved:
        remove_path(backup)


def ffmpeg_version(ffmpeg: Path) -> str:
    result = subprocess.run(
        [str(ffmpeg), "-version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    first = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not first.startswith("ffmpeg version "):
        raise RuntimeError("cannot authenticate FFmpeg version")
    return first


def _ffprobe_for_ffmpeg(ffmpeg: Path) -> Path:
    suffix = ffmpeg.suffix
    candidate = ffmpeg.with_name(f"ffprobe{suffix}")
    try:
        return candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"FFprobe must be installed beside FFmpeg: {candidate}"
        ) from error


def ffprobe_version(ffprobe: Path) -> str:
    result = subprocess.run(
        [str(ffprobe), "-version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    first = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not first.startswith("ffprobe version "):
        raise RuntimeError("cannot authenticate FFprobe version")
    return first


def _probe_frame_count(ffprobe: Path, source: Path) -> int:
    result = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", str(source),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    try:
        count = int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"cannot count source video frames: {source}") from error
    if count <= 0:
        raise RuntimeError(f"source video contains no frames: {source}")
    return count


def _validated_rows(chug_root: Path):
    selection_path = chug_root / "selection_manifest.json"
    receipt_path = chug_root / "download_receipt.json"
    selection = read_json(selection_path, "CHUG selection manifest")
    receipt = read_json(receipt_path, "CHUG download receipt")
    if (selection.get("schema") != 1 or receipt.get("schema") != 1 or
            receipt.get("license") != "CC BY-NC-SA 4.0"):
        raise RuntimeError("unsupported CHUG source/usage contract")
    accepted = receipt.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise RuntimeError("CHUG receipt contains no accepted native videos")
    rows = []
    seen_video = set()
    for row in accepted:
        if not isinstance(row, dict):
            raise RuntimeError("CHUG accepted row is invalid")
        video_id = row.get("video_id")
        split = row.get("split")
        audit = row.get("audit", {})
        download = row.get("download", {})
        video = chug_root / "videos" / f"{video_id}.mp4"
        if (not isinstance(video_id, str) or len(video_id) != 32 or
                video_id in seen_video or split not in {*SPLITS, "test"}):
            raise RuntimeError("CHUG accepted identity/split is invalid")
        seen_video.add(video_id)
        expected = {
            "codec": "hevc",
            "color_range": "tv",
            "color_primaries": "bt2020",
            "color_space": "bt2020nc",
            "color_transfer": "smpte2084",
        }
        mismatch = {
            key: (value, audit.get(key)) for key, value in expected.items()
            if audit.get(key) != value
        }
        if (mismatch or not str(audit.get("pixel_format", "")).startswith(
                "yuv420p10")):
            raise RuntimeError(f"{video_id}: native PQ stream differs: {mismatch}")
        if (row.get("orientation") != "Landscape" or
                int(audit.get("width", 0)) * 9 !=
                int(audit.get("height", 0)) * 16):
            continue
        if (not video.is_file() or download.get("sha256") != sha256(video) or
                download.get("bytes") != video.stat().st_size):
            raise RuntimeError(f"{video_id}: CHUG native-video identity differs")
        rows.append({**row, "video_path": video})
    selection_ids = {
        row.get("video_id") for row in selection.get("clips", [])
        if isinstance(row, dict)
    }
    if seen_video != selection_ids:
        raise RuntimeError("CHUG selection and download receipt identities differ")
    return rows, {
        "selection_manifest": str(selection_path.resolve()),
        "selection_manifest_sha256": sha256(selection_path),
        "download_receipt": str(receipt_path.resolve()),
        "download_receipt_sha256": sha256(receipt_path),
    }


def _select_unique(rows, split: str, count: int):
    candidates = [row for row in rows if row["split"] == split]
    by_group = {}
    for row in candidates:
        group = row.get("capture_group_id")
        if not isinstance(group, str) or len(group) != 64:
            raise RuntimeError("CHUG row lacks a capture-group identity")
        previous = by_group.get(group)
        key = stable_hash(SELECTION_SEED_CONTRACT, split, row["video_id"])
        if previous is None or key < previous[0]:
            by_group[group] = (key, row)
    unique = [item[1] for item in by_group.values()]
    buckets = {}
    for row in unique:
        buckets.setdefault(str(row.get("frame_rate_bucket")), []).append(row)
    for values in buckets.values():
        values.sort(key=lambda row: stable_hash(
            SELECTION_SEED_CONTRACT, split, row["capture_group_id"],
            row["video_id"],
        ))
    selected = []
    names = sorted(buckets)
    while len(selected) < count and any(buckets.values()):
        for name in names:
            if buckets[name] and len(selected) < count:
                selected.append(buckets[name].pop(0))
    if len(selected) != count:
        raise RuntimeError(
            f"CHUG has only {len(selected)} group-unique {split} landscape clips"
        )
    return selected


def pq_eotf_nits(rgb_pq: np.ndarray) -> np.ndarray:
    value = np.clip(rgb_pq, np.float32(0.0), np.float32(1.0))
    powered = np.power(value, np.float32(1.0 / PQ["m2"]))
    numerator = np.maximum(powered - np.float32(PQ["c1"]), np.float32(0.0))
    denominator = np.maximum(
        np.float32(PQ["c2"]) - np.float32(PQ["c3"]) * powered,
        np.float32(1e-12),
    )
    return (
        np.power(numerator / denominator, np.float32(1.0 / PQ["m1"])) *
        np.float32(PQ["peak_nits"])
    ).astype(np.float32, copy=False)


def nonlinear_bt2020_to_scrgb(rgb_pq: np.ndarray) -> np.ndarray:
    nits = pq_eotf_nits(rgb_pq)
    rec709_nits = np.matmul(nits, BT2020_TO_REC709.T)
    return (rec709_nits / np.float32(80.0)).astype(np.float32, copy=False)


def _model_preview(scrgb_f16: np.ndarray) -> np.ndarray:
    # This is the exact CPU mirror of DepthHdrScRgbToSrgb before ImageNet
    # normalization.  The FP16 input boundary has already happened.
    rgb = scrgb_f16[..., :3].astype(np.float32)
    rgb = np.maximum(rgb, np.float32(0.0))
    luminance = np.maximum(
        np.sum(rgb * REC709_LUMA, axis=2, keepdims=True, dtype=np.float32),
        np.float32(0.0),
    )
    rgb = rgb / (np.float32(1.0) + luminance)
    peak = np.max(rgb, axis=2, keepdims=True)
    rgb = rgb / np.maximum(peak, np.float32(1.0))
    rgb = np.clip(rgb, np.float32(0.0), np.float32(1.0))
    low = rgb * np.float32(12.92)
    high = (
        np.float32(1.055) * np.power(rgb, np.float32(1.0 / 2.4)) -
        np.float32(0.055)
    )
    return np.where(rgb <= np.float32(0.0031308), low, high).astype(
        np.float32, copy=False
    )


def _frame_stats(scrgb_f16: np.ndarray, preview: np.ndarray):
    rgb = scrgb_f16[..., :3].astype(np.float32)
    luminance_nits = np.maximum(
        np.sum(rgb * REC709_LUMA, axis=2, dtype=np.float32) * np.float32(80.0),
        np.float32(0.0),
    )
    percentiles = np.percentile(luminance_nits, (50, 95, 99, 99.9))
    return {
        "scrgb_component_min": float(np.min(rgb)),
        "scrgb_component_max": float(np.max(rgb)),
        "negative_component_fraction": float(np.mean(rgb < 0.0)),
        "superwhite_component_fraction": float(np.mean(rgb > 1.0)),
        "over_1000_nit_component_fraction": float(np.mean(rgb > 12.5)),
        "luminance_nits_p50": float(percentiles[0]),
        "luminance_nits_p95": float(percentiles[1]),
        "luminance_nits_p99": float(percentiles[2]),
        "luminance_nits_p999": float(percentiles[3]),
        "luminance_nits_max": float(np.max(luminance_nits)),
        "preview_black_fraction": float(np.mean(np.max(preview, axis=2) <= 0.0)),
        "preview_saturated_fraction": float(np.mean(np.max(preview, axis=2) >= 1.0)),
        "nonfinite_components": int(np.size(rgb) - np.isfinite(rgb).sum()),
    }


def _materialize_capture_frame(planar, source_width: int, source_height: int,
                               width: int, height: int):
    """Apply the exact persisted scRGB/preview conversion to one FFmpeg frame."""
    nonlinear = np.stack((planar[2], planar[0], planar[1]), axis=2)
    scrgb = nonlinear_bt2020_to_scrgb(nonlinear)
    # Mirror an FP16 compositor/capture boundary before the production
    # half-pixel sample, then materialize the canonical capture surface.
    scrgb = scrgb.astype(np.float16).astype(np.float32)
    if (source_width, source_height) != (width, height):
        scrgb = input_color._bilinear_resize_numpy(  # noqa: SLF001
            scrgb, width, height
        )
    rgb_f16 = scrgb.astype("<f2")
    rgba_f16 = np.empty((height, width, 4), dtype="<f2")
    rgba_f16[..., :3] = rgb_f16
    rgba_f16[..., 3] = np.float16(1.0)
    if not np.isfinite(rgba_f16).all():
        raise RuntimeError("non-finite scRGB derivative")
    preview = _model_preview(rgba_f16)
    preview_u8 = np.rint(
        np.clip(preview, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    return rgba_f16, preview, preview_u8


def _conversion_contract(ffmpeg: Path, version: str, ffprobe: Path,
                         probe_version: str, width: int, height: int):
    filter_text = (
        "zscale=matrixin=2020_ncl:matrix=gbr:rangein=limited:range=full:"
        "transferin=smpte2084:transfer=smpte2084:primariesin=2020:"
        "primaries=2020:filter=bilinear:dither=none,format=gbrpf32le"
    )
    payload = {
        "schema": CONVERSION_SCHEMA,
        "contract": CONVERSION_CONTRACT,
        "source": {
            "codec": "hevc",
            "pixel_format": "yuv420p10+",
            "range": "limited",
            "primaries": "bt2020",
            "matrix": "bt2020nc",
            "transfer": "smpte2084",
        },
        "decoder": {
            "path": str(ffmpeg.resolve()),
            "sha256": sha256(ffmpeg),
            "version": version,
            "threads": 1,
            "output_pixel_format": "gbrpf32le",
            "filter": filter_text,
            "role": "HEVC decode, chroma reconstruction, and YUV-to-nonlinear-RGB only",
        },
        "frame_counter": {
            "path": str(ffprobe.resolve()),
            "sha256": sha256(ffprobe),
            "version": probe_version,
            "mode": "video-stream-0-packet-count",
        },
        "pq_eotf": PQ,
        "linear_bt2020_to_linear_rec709_d65": BT2020_TO_REC709.tolist(),
        "scrgb_reference_white_nits": 80.0,
        "source_fp16_boundary": "before-half-pixel-bilinear-resize",
        "resize": {
            "width": width,
            "height": height,
            "filter": "half-pixel bilinear",
            "address_mode": "clamp",
        },
        "capture_store": "little-endian RGBA16F; alpha=1",
        "preview": native_hdr_capture.PREVIEW_ENCODING,
        "depth_input_color_contract_sha256": input_color.color_contract_sha256(),
    }
    return payload


def _decode_command(ffmpeg: Path, source: Path, filter_text: str,
                    source_frame_ids):
    if not source_frame_ids:
        raise RuntimeError("native-HDR sparse frame selection is empty")
    selector = "+".join(
        f"eq(n\\,{frame_id})" for frame_id in source_frame_ids
    )
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "1", "-i", str(source), "-map", "0:v:0", "-an",
        "-sn", "-dn", "-vf", f"select={selector},{filter_text}",
        "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ]


def _flow_search_offsets(radius: int):
    """Return the frozen nearest-center order: 0, -1, +1, -2, +2, ..."""
    if type(radius) is not int or radius < 0:
        raise RuntimeError("flow-support search radius must be nonnegative")
    offsets = [0]
    for distance in range(1, radius + 1):
        offsets.extend((-distance, distance))
    return offsets


def _decode_flow_previews(row, ffmpeg: Path, conversion, width: int,
                          height: int, source_frame_ids):
    """Decode exact persisted-preview luma at the evaluator's flow raster."""
    selected_ids = sorted(set(source_frame_ids))
    if not selected_ids:
        return {}
    source_width = int(row["audit"]["width"])
    source_height = int(row["audit"]["height"])
    frame_bytes = source_width * source_height * 3 * 4
    command = _decode_command(
        ffmpeg, row["video_path"], conversion["decoder"]["filter"],
        selected_ids,
    )
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RuntimeError("cannot open FFmpeg flow-curation decode pipes")
    previews = {}
    try:
        output_frame = 0
        while True:
            chunks = bytearray()
            while len(chunks) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if not chunks:
                break
            if len(chunks) != frame_bytes:
                raise RuntimeError(
                    f"{row['video_id']}: FFmpeg returned a truncated flow frame"
                )
            if output_frame >= len(selected_ids):
                raise RuntimeError(
                    f"{row['video_id']}: FFmpeg returned extra flow frames"
                )
            planar = np.frombuffer(chunks, dtype="<f4").reshape(
                3, source_height, source_width
            )
            _rgba, _preview, preview_u8 = _materialize_capture_frame(
                planar, source_width, source_height, width, height
            )
            preview_rgb = preview_u8.astype(np.float32) / np.float32(255.0)
            gray = (
                preview_rgb[..., 0] * np.float32(0.2126) +
                preview_rgb[..., 1] * np.float32(0.7152) +
                preview_rgb[..., 2] * np.float32(0.0722)
            )
            flow_width = min(256, width)
            flow_height = max(24, round(height * flow_width / width))
            previews[selected_ids[output_frame]] = sbsbench.resize_to(
                gray, flow_width, flow_height
            )
            output_frame += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=60)
        process = None
        if return_code != 0:
            raise RuntimeError(
                f"{row['video_id']}: FFmpeg flow decode failed "
                f"({return_code}): {stderr[-1000:]}"
            )
        if output_frame != len(selected_ids):
            raise RuntimeError(
                f"{row['video_id']}: FFmpeg selected {output_frame} flow "
                f"frames; expected {len(selected_ids)}"
            )
        return previews
    finally:
        if process is not None:
            process.kill()
            process.wait()


def _previous_pair_support(previews, center: int):
    try:
        previous = previews[center - 1]
        current = previews[center]
    except KeyError as error:
        raise RuntimeError(
            f"missing flow-curation preview for center {center}"
        ) from error
    height, width = current.shape
    if previous.shape != current.shape:
        raise RuntimeError("flow-curation preview dimensions differ")
    _temporal, _depth, support = sbsbench.flow_temporal_metrics(
        current, current, previous, previous, current, previous,
        min_support=1.1,
    )
    return support


def _curated_frame_selection_plan(row, ffmpeg: Path, conversion, width: int,
                                  height: int, labels_per_clip: int):
    """Move only unsupported nominal centers using a bounded frozen search.

    The evaluator always prefers target-1 -> target when that pair exists.  A
    window is therefore admissible only when that exact source pair reaches the
    evaluator's minimum flow-support threshold.  The selected support and the
    full deterministic search contract are persisted in clip metadata and then
    covered by the clip-hash manifest.
    """
    source_frame_count = row["source_frame_count"]
    initial = _frame_selection_plan(source_frame_count, labels_per_clip)
    candidate_centers = {}
    initial_ids = set()
    for window in initial["windows"]:
        nominal = window["source_label_frame_id"]
        candidates = [
            nominal + offset
            for offset in _flow_search_offsets(FLOW_SUPPORT_SEARCH_RADIUS_FRAMES)
            if 1 <= nominal + offset <= source_frame_count - 2
        ]
        candidate_centers[nominal] = candidates
        initial_ids.update((nominal - 1, nominal))

    previews = _decode_flow_previews(
        row, ffmpeg, conversion, width, height, initial_ids
    )
    initial_support = {
        nominal: _previous_pair_support(previews, nominal)
        for nominal in candidate_centers
    }
    failed_nominals = [
        nominal for nominal, support in initial_support.items()
        if support < FLOW_TEMPORAL_MIN_SUPPORT
    ]
    if failed_nominals:
        search_ids = set()
        for nominal in failed_nominals:
            for center in candidate_centers[nominal]:
                search_ids.update((center - 1, center))
        missing = search_ids - set(previews)
        previews.update(_decode_flow_previews(
            row, ffmpeg, conversion, width, height, missing
        ))

    selected_windows = []
    selected_centers = []
    for window in initial["windows"]:
        nominal = window["source_label_frame_id"]
        selected_center = None
        selected_support = None
        for center in candidate_centers[nominal]:
            if selected_centers and center <= selected_centers[-1]:
                continue
            support = (
                initial_support[nominal] if center == nominal else
                _previous_pair_support(previews, center)
            )
            if support >= FLOW_TEMPORAL_MIN_SUPPORT:
                selected_center = center
                selected_support = support
                break
        if selected_center is None:
            raise RuntimeError(
                f"{row['video_id']}: no flow-valid center within "
                f"+/-{FLOW_SUPPORT_SEARCH_RADIUS_FRAMES} frames of {nominal}"
            )
        selected_centers.append(selected_center)
        selected_window = dict(window)
        selected_window["source_label_frame_id"] = selected_center
        selected_window["source_frame_ids"] = [
            selected_center - 1, selected_center, selected_center + 1
        ]
        selected_window["temporal_evidence_selection"] = {
            "contract": FLOW_SUPPORT_SELECTION_CONTRACT,
            "flow_support_contract": FLOW_SUPPORT_CONTRACT,
            "flow_support_metric_sha256": flow_support_metric_sha256(),
            "preferred_pair": "previous-source-frame-to-label-frame",
            "minimum_support": FLOW_TEMPORAL_MIN_SUPPORT,
            "search_radius_frames": FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
            "search_order": "nominal-then-negative-positive-by-distance",
            "nominal_source_label_frame_id": nominal,
            "selected_source_label_frame_id": selected_center,
            "selected_offset_frames": selected_center - nominal,
            "selected_previous_source_frame_id": selected_center - 1,
            "selected_pair_flow_support": float(selected_support),
        }
        selected_windows.append(selected_window)

    retained = sorted({
        frame_id for window in selected_windows
        for frame_id in window["source_frame_ids"]
    })
    plan = dict(initial)
    plan.update({
        "flow_support_selection_contract": FLOW_SUPPORT_SELECTION_CONTRACT,
        "flow_support_contract": FLOW_SUPPORT_CONTRACT,
        "flow_support_metric_sha256": flow_support_metric_sha256(),
        "flow_support_minimum": FLOW_TEMPORAL_MIN_SUPPORT,
        "flow_support_search_radius_frames": FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
        "flow_support_search_order": (
            "nominal-then-negative-positive-by-distance"
        ),
        "initial_source_label_frame_ids": initial["source_label_frame_ids"],
        "source_label_frame_ids": selected_centers,
        "source_frame_ids": retained,
        "windows": selected_windows,
    })
    return plan


def _label_frame_ids(frame_count: int, count: int):
    if frame_count < count + 2:
        raise RuntimeError("native-HDR clip is too short for sparse temporal labels")
    values = []
    for index in range(count):
        value = int(round((index + 1) * (frame_count - 1) / (count + 1)))
        value = min(max(value, 1), frame_count - 2)
        values.append(value)
    values = sorted(set(values))
    if len(values) != count:
        raise RuntimeError("native-HDR sparse label identities collapsed")
    return values


def _receipt_frame_count(row) -> int:
    try:
        duration = float(row["probed_duration_seconds"])
        fps = float(row["probed_frame_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("CHUG row has invalid duration/frame rate") from error
    if (not math.isfinite(duration) or duration <= 0.0 or
            not math.isfinite(fps) or fps <= 0.0):
        raise RuntimeError("CHUG row has invalid duration/frame rate")
    count = int(round(duration * fps))
    if count <= 0:
        raise RuntimeError("CHUG row resolves to no source frames")
    return count


def _frame_selection_plan(source_frame_count: int, labels_per_clip: int):
    source_labels = _label_frame_ids(source_frame_count, labels_per_clip)
    windows = []
    for window_index, source_label in enumerate(source_labels):
        source_ids = list(range(
            source_label - TEMPORAL_WINDOW_RADIUS,
            source_label + TEMPORAL_WINDOW_RADIUS + 1,
        ))
        windows.append({
            "window_index": window_index,
            "source_frame_ids": source_ids,
            "source_label_frame_id": source_label,
            "frame_ids": list(range(len(source_ids))),
            "label_frame_ids": [TEMPORAL_WINDOW_RADIUS],
        })
    retained = sorted({
        frame_id for window in windows
        for frame_id in window["source_frame_ids"]
    })
    if (not retained or retained[0] < 0 or
            retained[-1] >= source_frame_count):
        raise RuntimeError("native-HDR temporal window escapes its source clip")
    return {
        "contract": FRAME_SELECTION_CONTRACT,
        "source_frame_count": source_frame_count,
        "temporal_window_radius": TEMPORAL_WINDOW_RADIUS,
        "source_frame_ids": retained,
        "source_label_frame_ids": source_labels,
        "windows": windows,
    }


def _frame_selection_metadata(plan, window, fps: float):
    metadata = {
        "contract": plan["contract"],
        "source_frame_count": plan["source_frame_count"],
        "source_frame_rate": fps,
        "window_index": window["window_index"],
        "retained_frame_count": len(window["frame_ids"]),
        "temporal_window_radius": plan["temporal_window_radius"],
        "label_frame_ids": window["label_frame_ids"],
        "source_label_frame_id": window["source_label_frame_id"],
        "frames": [
            {
                "frame": local_frame,
                "source_frame": source_frame,
                "source_timestamp_seconds": source_frame / fps,
            }
            for local_frame, source_frame in zip(
                window["frame_ids"], window["source_frame_ids"]
            )
        ],
    }
    evidence = window.get("temporal_evidence_selection")
    if evidence is not None:
        metadata["temporal_evidence_selection"] = evidence
    return metadata


def _retention_summary(selected, width: int, height: int,
                       labels_per_clip: int):
    split_rows = {}
    total_source = 0
    total_retained = 0
    for split, rows in selected.items():
        source_frames = sum(row["source_frame_count"] for row in rows)
        retained_frames = sum(
            sum(len(window["frame_ids"]) for window in row.get(
                "frame_selection",
                _frame_selection_plan(
                    row["source_frame_count"], labels_per_clip
                ),
            )["windows"])
            for row in rows
        )
        split_rows[split] = {
            "source_clips": len(rows),
            "window_clips": len(rows) * labels_per_clip,
            "source_frames": source_frames,
            "retained_frames": retained_frames,
            "label_frames": len(rows) * labels_per_clip,
        }
        total_source += source_frames
        total_retained += retained_frames
    bytes_per_frame = width * height * SCRGB_BYTES_PER_PIXEL
    full_bytes = total_source * bytes_per_frame
    retained_bytes = total_retained * bytes_per_frame
    return {
        "contract": FRAME_SELECTION_CONTRACT,
        "temporal_window_radius": TEMPORAL_WINDOW_RADIUS,
        "temporal_evidence_selection": {
            "contract": FLOW_SUPPORT_SELECTION_CONTRACT,
            "flow_support_contract": FLOW_SUPPORT_CONTRACT,
            "flow_support_metric_sha256": flow_support_metric_sha256(),
            "minimum_support": FLOW_TEMPORAL_MIN_SUPPORT,
            "search_radius_frames": FLOW_SUPPORT_SEARCH_RADIUS_FRAMES,
            "search_order": "nominal-then-negative-positive-by-distance",
            "preferred_pair": "previous-source-frame-to-label-frame",
        },
        "stored_identity": (
            "independent-contiguous-window-clip-with-source-frame-map"
        ),
        "splits": split_rows,
        "total": {
            "source_frames": total_source,
            "retained_frames": total_retained,
            "discarded_frames": total_source - total_retained,
            "retained_fraction": total_retained / total_source,
            "raw_scrgb16_full_bytes": full_bytes,
            "raw_scrgb16_full_gib": full_bytes / (1024 ** 3),
            "raw_scrgb16_retained_bytes": retained_bytes,
            "raw_scrgb16_retained_gib": retained_bytes / (1024 ** 3),
            "raw_scrgb16_saved_bytes": full_bytes - retained_bytes,
            "raw_scrgb16_saved_gib": (
                (full_bytes - retained_bytes) / (1024 ** 3)
            ),
        },
    }


def _window_clip_name(video_id: str, window_index: int) -> str:
    return f"chug_pq_{video_id}_w{window_index:02d}"


def _window_result(row, clip_name: str, window, status: str):
    evidence = window.get("temporal_evidence_selection")
    return {
        "clip": clip_name,
        "status": status,
        "frames": len(window["frame_ids"]),
        "source_frames": len(window["source_frame_ids"]),
        "master_source_frames": row["source_frame_count"],
        "source_frame_rate": float(row["probed_frame_rate"]),
        "label_frames": len(window["label_frame_ids"]),
        "capture_group_id": row["capture_group_id"],
        "video_id": row["video_id"],
        "window_index": window["window_index"],
        "source_label_frame_id": window["source_label_frame_id"],
        **({
            "nominal_source_label_frame_id": evidence[
                "nominal_source_label_frame_id"
            ],
            "selected_pair_flow_support": evidence[
                "selected_pair_flow_support"
            ],
            "temporal_evidence_selection": evidence,
        } if evidence is not None else {}),
    }


def _prepare_clip(row, split_root: Path, ffmpeg: Path, conversion,
                  conversion_hash: str, width: int, height: int,
                  labels_per_clip: int, overwrite: bool, selection):
    video_id = row["video_id"]
    source_frame_count = row.get("source_frame_count")
    if type(source_frame_count) is not int or source_frame_count <= 0:
        raise RuntimeError(f"{video_id}: invalid authenticated source frame count")
    fps = float(row["probed_frame_rate"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"{video_id}: invalid probed frame rate")
    if (not isinstance(selection, dict) or
            selection.get("flow_support_selection_contract") !=
            FLOW_SUPPORT_SELECTION_CONTRACT or
            not isinstance(selection.get("windows"), list) or
            not selection.get("windows") or
            any(not isinstance(window, dict) or
                window.get("temporal_evidence_selection", {}).get("contract") !=
                FLOW_SUPPORT_SELECTION_CONTRACT
                for window in selection.get("windows", []))):
        raise RuntimeError(
            f"{video_id}: missing authenticated flow-curated frame selection"
        )
    states = []
    for window in selection["windows"]:
        clip_name = _window_clip_name(video_id, window["window_index"])
        states.append({
            "clip_name": clip_name,
            "destination": split_root / clip_name,
            "window": window,
            "selection_metadata": _frame_selection_metadata(
                selection, window, fps
            ),
        })

    reusable = []
    if not overwrite:
        for state in states:
            destination = state["destination"]
            try:
                authentication = native_hdr_capture.validate_clip(
                    destination, full=False
                )
                payload = read_json(
                    destination / native_hdr_capture.MANIFEST_NAME,
                    "native-HDR frame manifest",
                )
                labels = read_json(
                    destination / "label_frames.json", "label frames"
                )
                metadata = read_json(
                    destination / "meta.json", "clip metadata"
                )
                valid = (
                    payload["source_video"]["sha256"] ==
                    row["download"]["sha256"] and
                    payload["conversion"]["contract_sha256"] ==
                    conversion_hash and
                    authentication["frame_count"] ==
                    len(state["window"]["frame_ids"]) and
                    labels == {
                        "schema": 1,
                        "frame_ids": state["window"]["label_frame_ids"],
                    } and
                    metadata.get("preparation_contract") ==
                    PREPARATION_CONTRACT and
                    metadata.get("frame_selection") ==
                    state["selection_metadata"]
                )
            except (KeyError, OSError, RuntimeError, ValueError):
                valid = False
            reusable.append(valid)
    if reusable and all(reusable):
        return [
            _window_result(
                row, state["clip_name"], state["window"], "reused"
            )
            for state in states
        ]

    split_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(
        prefix=f".chug_pq_{video_id}.partial-", dir=split_root
    ))
    process = None
    try:
        occurrences = {}
        for state in states:
            staging = staging_root / state["clip_name"]
            model_root = staging / native_hdr_capture.MODEL_SOURCE_DIRECTORY
            model_root.mkdir(parents=True)
            state["staging"] = staging
            state["model_root"] = model_root
            state["records"] = []
            for local_frame, source_frame in zip(
                    state["window"]["frame_ids"],
                    state["window"]["source_frame_ids"]):
                occurrences.setdefault(source_frame, []).append(
                    (state, local_frame)
                )

        source_width = int(row["audit"]["width"])
        source_height = int(row["audit"]["height"])
        frame_bytes = source_width * source_height * 3 * 4
        command = _decode_command(
            ffmpeg, row["video_path"], conversion["decoder"]["filter"],
            selection["source_frame_ids"],
        )
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("cannot open FFmpeg native-HDR decode pipes")
        output_frame = 0
        while True:
            chunks = bytearray()
            while len(chunks) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if not chunks:
                break
            if len(chunks) != frame_bytes:
                raise RuntimeError(
                    f"{video_id}: FFmpeg returned a truncated float frame"
                )
            if output_frame >= len(selection["source_frame_ids"]):
                raise RuntimeError(
                    f"{video_id}: FFmpeg returned extra selected frames"
                )
            source_frame_id = selection["source_frame_ids"][output_frame]
            planar = np.frombuffer(chunks, dtype="<f4").reshape(
                3, source_height, source_width
            )
            try:
                rgba_f16, preview, preview_u8 = _materialize_capture_frame(
                    planar, source_width, source_height, width, height
                )
            except RuntimeError as error:
                raise RuntimeError(f"{video_id}: {error}") from error
            stats = _frame_stats(rgba_f16, preview)
            for state, frame_id in occurrences[source_frame_id]:
                if frame_id != len(state["records"]):
                    raise RuntimeError(
                        f"{state['clip_name']}: window cadence is not contiguous"
                    )
                suffix = f"{frame_id:05d}"
                model_path = (
                    state["model_root"] / f"frame_{suffix}.scrgb16"
                )
                rgba_f16.tofile(model_path)
                model_stat = model_path.stat()
                preview_path = state["staging"] / f"frame_{suffix}.png"
                Image.fromarray(preview_u8, "RGB").save(
                    preview_path, format="PNG", compress_level=6,
                    optimize=False
                )
                state["records"].append({
                    "frame": frame_id,
                    "path": (
                        f"{native_hdr_capture.MODEL_SOURCE_DIRECTORY}/"
                        f"frame_{suffix}.scrgb16"
                    ),
                    "size": model_stat.st_size,
                    "mtime_ns": model_stat.st_mtime_ns,
                    "sha256": sha256(model_path),
                    "preview": f"frame_{suffix}.png",
                    "preview_sha256": sha256(preview_path),
                    "timestamp_seconds": source_frame_id / fps,
                    "stats": stats,
                })
            output_frame += 1
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=60)
        process = None
        if return_code != 0:
            raise RuntimeError(
                f"{video_id}: FFmpeg decode failed ({return_code}): "
                f"{stderr[-1000:]}"
            )
        if output_frame != len(selection["source_frame_ids"]):
            raise RuntimeError(
                f"{video_id}: FFmpeg selected {output_frame} frames; "
                f"expected {len(selection['source_frame_ids'])}"
            )

        for state in states:
            window = state["window"]
            records = state["records"]
            if len(records) != len(window["frame_ids"]):
                raise RuntimeError(
                    f"{state['clip_name']}: incomplete temporal window"
                )
            source_video = {
                "dataset": "CHUG",
                "video_id": video_id,
                "path": str(row["video_path"].resolve()),
                "bytes": row["download"]["bytes"],
                "sha256": row["download"]["sha256"],
                "split": row["split"],
                "capture_group_id": row["capture_group_id"],
                "content_id": row["content_id"],
                "license": "CC BY-NC-SA 4.0",
                "source_frame_count": source_frame_count,
                "source_frame_rate": fps,
                "frame_selection_contract": FRAME_SELECTION_CONTRACT,
                "window_index": window["window_index"],
                "source_window_frame_ids": window["source_frame_ids"],
                "source_label_frame_id": window["source_label_frame_id"],
                "source_window_timestamps_seconds": [
                    value / fps for value in window["source_frame_ids"]
                ],
                **({
                    "temporal_evidence_selection": window[
                        "temporal_evidence_selection"
                    ],
                } if "temporal_evidence_selection" in window else {}),
            }
            conversion_identity = {
                "contract": CONVERSION_CONTRACT,
                "contract_sha256": conversion_hash,
            }
            semantic = {
                "contract": native_hdr_capture.MANIFEST_CONTRACT,
                "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
                "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
                "width": width,
                "height": height,
                "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
                "source_video": source_video,
                "conversion": conversion_identity,
                "frames": [{
                    key: value for key, value in record.items()
                    if key != "mtime_ns" and key != "stats"
                } for record in records],
            }
            frame_manifest = {
                "schema": native_hdr_capture.MANIFEST_SCHEMA,
                "contract": native_hdr_capture.MANIFEST_CONTRACT,
                "capture_encoding": native_hdr_capture.CAPTURE_ENCODING,
                "preview_encoding": native_hdr_capture.PREVIEW_ENCODING,
                "width": width,
                "height": height,
                "row_pitch_bytes": width * SCRGB_BYTES_PER_PIXEL,
                "source_video": source_video,
                "conversion": conversion_identity,
                "frames": records,
                "frame_count": len(records),
                "content_sha256": native_hdr_capture.canonical_sha256(
                    semantic
                ),
            }
            write_json_atomic(
                state["staging"] / native_hdr_capture.MANIFEST_NAME,
                frame_manifest,
            )
            write_json_atomic(state["staging"] / "label_frames.json", {
                "schema": 1,
                "frame_ids": state["window"]["label_frame_ids"],
            })
            write_json_atomic(state["staging"] / "meta.json", {
                "name": (
                    f"CHUG native PQ {video_id} window "
                    f"{state['window']['window_index']:02d}"
                ),
                "description": (
                    "Authenticated CHUG BT.2020/PQ three-frame window "
                    "converted to FP16 Windows scRGB"
                ),
                "dataset": "chug",
                "production_id": f"chug_native_pq_v1_{row['split']}",
                "split": row["split"],
                "source_kind": "native-hdr-video",
                "license": "CC BY-NC-SA 4.0",
                "capture_group_id": row["capture_group_id"],
                "source_video_id": video_id,
                "native_hdr": True,
                "required_gt_depth": False,
                "required_gt_flow": False,
                "required_gt_stereo": False,
                "global_policy_weight": 1.0,
                "preparation_contract": PREPARATION_CONTRACT,
                "frame_selection": state["selection_metadata"],
            })
            native_hdr_capture.validate_clip(state["staging"], full=True)

        for state in states:
            publish_directory(state["staging"], state["destination"])
        return [
            _window_result(
                row, state["clip_name"], state["window"], "prepared"
            )
            for state in states
        ]
    finally:
        if process is not None:
            process.kill()
            process.wait()
        remove_path(staging_root)


def _dataset_manifest(split: str, rows, preparation, conversion_hash: str):
    sequence_rows = sorted(rows, key=lambda row: row["clip"])
    master_frames = {}
    for row in sequence_rows:
        previous = master_frames.setdefault(
            row["video_id"], row["master_source_frames"]
        )
        if previous != row["master_source_frames"]:
            raise RuntimeError("CHUG source frame count changed across windows")
    return {
        "schema": 2,
        "dataset": "chug-native-pq-v1",
        "domain": "native_hdr_cinematic",
        "production_id": f"chug_native_pq_v1_{split}",
        "source_kind": "native-hdr-video",
        "split": split,
        "source_split": split,
        "projection": "rectilinear",
        "policy_role": "cinematic_training",
        "global_policy_weight": 1.0,
        "license": "CC BY-NC-SA 4.0",
        "preparation_contract": PREPARATION_CONTRACT,
        "temporal_evidence_selection_contract": (
            FLOW_SUPPORT_SELECTION_CONTRACT
        ),
        "source_flow_support_contract": FLOW_SUPPORT_CONTRACT,
        "source_flow_metric_sha256": flow_support_metric_sha256(),
        "source_flow_support_minimum": FLOW_TEMPORAL_MIN_SUPPORT,
        "window_grouping_contract": (
            "all-windows-inherit-source-video-capture-group-and-split"
        ),
        "conversion_contract_sha256": conversion_hash,
        "sequences": [{
            "clip": row["clip"],
            "frames": row["frames"],
            "source_frames": row["source_frames"],
            "master_source_frames": row["master_source_frames"],
            "source_frame_rate": row["source_frame_rate"],
            "label_frames": row["label_frames"],
            "split": split,
            "capture_group_id": row["capture_group_id"],
            "video_id": row["video_id"],
            "window_index": row["window_index"],
            "source_label_frame_id": row["source_label_frame_id"],
            **({
                "nominal_source_label_frame_id": row[
                    "nominal_source_label_frame_id"
                ],
                "selected_pair_flow_support": row[
                    "selected_pair_flow_support"
                ],
                "temporal_evidence_selection": row[
                    "temporal_evidence_selection"
                ],
            } if "temporal_evidence_selection" in row else {}),
        } for row in sequence_rows],
        "frame_count": sum(row["frames"] for row in sequence_rows),
        "window_clip_count": len(sequence_rows),
        "source_video_count": len(master_frames),
        "source_frame_count": sum(
            row["source_frames"] for row in sequence_rows
        ),
        "master_source_frame_count": sum(master_frames.values()),
        "label_frame_count": sum(row["label_frames"] for row in sequence_rows),
        "source_provenance": preparation,
    }


def _build_clip_hash_manifest(split_root: Path, workers: int):
    # Import lazily so the preparation module remains independently testable.
    import sys
    sbsbench = Path(__file__).resolve().parents[1] / "sbsbench"
    sys.path.insert(0, str(sbsbench))
    import build_clip_hash_manifest as clip_hashes  # noqa: E402
    manifest, path = clip_hashes.build_and_write(
        split_root, workers=min(workers, 8)
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "semantic_content_sha256": manifest[
            clip_hashes.MANIFEST_CONTENT_SHA256_FIELD
        ],
    }


def prepare(args):
    chug_root = args.chug_root.resolve(strict=True)
    output_root = args.output_root.resolve(strict=False)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    if output_root == chug_root or output_root.is_relative_to(chug_root):
        raise RuntimeError("native-HDR output must not overlap CHUG masters")
    rows, source_provenance = _validated_rows(chug_root)
    selected = {
        "training": _select_unique(rows, "training", args.training_clips),
        "development": _select_unique(
            rows, "development", args.development_clips
        ),
    }
    groups = {
        row["capture_group_id"] for values in selected.values() for row in values
    }
    if len(groups) != args.training_clips + args.development_clips:
        raise RuntimeError("native-HDR capture group crosses the active split")
    ffprobe = _ffprobe_for_ffmpeg(ffmpeg)
    version = ffmpeg_version(ffmpeg)
    probe_version = ffprobe_version(ffprobe)
    conversion = _conversion_contract(
        ffmpeg, version, ffprobe, probe_version, args.width, args.height
    )
    conversion_hash = native_hdr_capture.canonical_sha256(conversion)
    for values in selected.values():
        for row in values:
            source_frame_count = _probe_frame_count(
                ffprobe, row["video_path"]
            )
            receipt_frame_count = _receipt_frame_count(row)
            if source_frame_count != receipt_frame_count:
                raise RuntimeError(
                    f"{row['video_id']}: packet count {source_frame_count} "
                    f"differs from receipt cadence {receipt_frame_count}"
                )
            row["source_frame_count"] = source_frame_count
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _curated_frame_selection_plan, row, ffmpeg, conversion,
                args.width, args.height, args.labels_per_clip,
            ): row
            for values in selected.values() for row in values
        }
        for future in as_completed(futures):
            row = futures[future]
            row["frame_selection"] = future.result()
    retention = _retention_summary(
        selected, args.width, args.height, args.labels_per_clip
    )
    if args.dry_run:
        return {
            "schema": PREPARATION_SCHEMA,
            "contract": PREPARATION_CONTRACT,
            "dry_run": True,
            "output_root": str(output_root),
            "conversion_contract": conversion,
            "conversion_contract_sha256": conversion_hash,
            "temporal_evidence_selection_contract": (
                FLOW_SUPPORT_SELECTION_CONTRACT
            ),
            "source_flow_support_contract": (
                FLOW_SUPPORT_CONTRACT
            ),
            "source_flow_metric_sha256": flow_support_metric_sha256(),
            "selected": {
                split: [{
                    "video_id": row["video_id"],
                    "capture_group_id": row["capture_group_id"],
                    "frame_rate": row["probed_frame_rate"],
                    "duration_seconds": row["probed_duration_seconds"],
                    "source_frame_count": row["source_frame_count"],
                    "frame_selection": row["frame_selection"],
                    "source_sha256": row["download"]["sha256"],
                } for row in values]
                for split, values in selected.items()
            },
            "retention": retention,
        }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "conversion_contract.json", conversion)
    prepared = {}
    for split in SPLITS:
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _prepare_clip, row, split_root, ffmpeg, conversion,
                    conversion_hash, args.width, args.height,
                    args.labels_per_clip, args.overwrite,
                    row["frame_selection"],
                ): row["video_id"]
                for row in selected[split]
            }
            for future in as_completed(futures):
                window_results = future.result()
                results.extend(window_results)
                for result in window_results:
                    print(
                        f"[{split}] {result['clip']} {result['status']} "
                        f"({result['frames']} frames)",
                        flush=True,
                    )
        results.sort(key=lambda row: row["clip"])
        manifest = _dataset_manifest(
            split, results, source_provenance, conversion_hash
        )
        manifest_path = split_root / "dataset_manifest.json"
        write_json_atomic(manifest_path, manifest)
        clip_hash = _build_clip_hash_manifest(split_root, args.workers)
        prepared[split] = {
            "root": str(split_root.resolve()),
            "dataset_manifest": str(manifest_path.resolve()),
            "dataset_manifest_sha256": sha256(manifest_path),
            "clip_hash_manifest": clip_hash,
            "clips": [row["clip"] for row in results],
            "context_frame_count": manifest["frame_count"],
            "source_context_frame_count": manifest["source_frame_count"],
            "master_source_frame_count": manifest[
                "master_source_frame_count"
            ],
            "label_frame_count": manifest["label_frame_count"],
            "capture_group_ids": sorted({
                row["capture_group_id"] for row in results
            }),
        }
    payload = {
        "schema": PREPARATION_SCHEMA,
        "contract": PREPARATION_CONTRACT,
        "output_root": str(output_root),
        "source_provenance": source_provenance,
        "conversion_contract": str(
            (output_root / "conversion_contract.json").resolve()
        ),
        "conversion_contract_sha256": conversion_hash,
        "temporal_evidence_selection_contract": (
            FLOW_SUPPORT_SELECTION_CONTRACT
        ),
        "source_flow_support_contract": FLOW_SUPPORT_CONTRACT,
        "source_flow_metric_sha256": flow_support_metric_sha256(),
        "source_flow_support_minimum": FLOW_TEMPORAL_MIN_SUPPORT,
        "sealed_test_policy": "CHUG test masters were not decoded or opened",
        "retention": retention,
        "datasets": prepared,
        "summary": {
            "training_clips": len(prepared["training"]["clips"]),
            "development_clips": len(prepared["development"]["clips"]),
            "training_policy_samples": prepared["training"][
                "label_frame_count"
            ],
            "development_policy_samples": prepared["development"][
                "label_frame_count"
            ],
        },
    }
    write_json_atomic(output_root / BOOTSTRAP_MANIFEST, payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chug-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", required=True, type=Path)
    parser.add_argument("--training-clips", type=int, default=TRAINING_COUNT)
    parser.add_argument(
        "--development-clips", type=int, default=DEVELOPMENT_COUNT
    )
    parser.add_argument("--labels-per-clip", type=int, default=LABELS_PER_CLIP)
    parser.add_argument("--width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for value, name, maximum in (
            (args.training_clips, "training clips", 72),
            (args.development_clips, "development clips", 12),
            (args.labels_per_clip, "labels per clip", 32),
            (args.width, "width", 8192),
            (args.height, "height", 8192),
            (args.workers, "workers", 4)):
        if type(value) is not int or value < 1 or value > maximum:
            parser.error(f"{name} must be between 1 and {maximum}")
    payload = prepare(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
