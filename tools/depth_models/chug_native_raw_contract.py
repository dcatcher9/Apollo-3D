#!/usr/bin/env python3
"""Stable producer contract for CHUG native-HDR raw frame CAS objects.

Only color conversion, FP16/preview materialization, timing normalization, and
threshold-independent cut signatures belong to this boundary.  Final dataset
names, label IDs, cut thresholds, sidecars, and reports intentionally do not.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import subprocess

import numpy as np
import PIL
from PIL import Image, ImageFile, PngImagePlugin, _binary

import native_hdr_capture
import native_runtime_identity as native_runtime
import prepare_chug_native_hdr_training as sparse
import preprocessing_artifact_cache as artifact_cache


SCHEMA = 1
CONTRACT = "apollo-chug-native-pq-raw-frames-v1"
MANIFEST = "raw_native_hdr_frames.json"


def _numpy_percentile_source():
    function = getattr(np.percentile, "__wrapped__", np.percentile)
    path = inspect.getsourcefile(function)
    if path is None:
        raise RuntimeError("cannot locate NumPy percentile implementation")
    return Path(path)


def runtime_identity(*, fresh=False):
    return {
        "numpy_version": np.__version__,
        "pillow_version": PIL.__version__,
        "pillow_core_version": getattr(
            Image.core, "PILLOW_VERSION", PIL.__version__
        ),
        "pillow_zlib_version": getattr(Image.core, "zlib_version", None),
        "float_contract": "little-endian-float16-scrgb-rgba",
        "preview_writer": "pillow-png-compress6-no-optimize",
        "native_binaries": {
            "numpy": native_runtime.module_native_identity(
                "numpy", np, fresh=fresh
            ),
            "pillow": native_runtime.module_native_identity(
                "pillow", PIL, fresh=fresh
            ),
        },
        "python_implementations": {
            "numpy_percentile": native_runtime.python_file_identity(
                "numpy-percentile", _numpy_percentile_source()
            ),
            "pillow_image": native_runtime.python_file_identity(
                "pillow-image", Image.__file__
            ),
            "pillow_image_file": native_runtime.python_file_identity(
                "pillow-image-file", ImageFile.__file__
            ),
            "pillow_png": native_runtime.python_file_identity(
                "pillow-png", PngImagePlugin.__file__
            ),
            "pillow_binary": native_runtime.python_file_identity(
                "pillow-binary", _binary.__file__
            ),
        },
    }


def verify_runtime_identity(expected):
    if runtime_identity(fresh=True) != expected:
        raise RuntimeError(
            "native-HDR preprocessing runtime changed during generation"
        )


def code_paths():
    return {
        "native_raw_contract": Path(__file__).resolve(),
        "native_hdr_capture": Path(native_hdr_capture.__file__).resolve(),
        "depth_input_color": Path(sparse.input_color.__file__).resolve(),
        "native_runtime_identity": Path(native_runtime.__file__).resolve(),
    }


def _materializer_source_identity():
    functions = (
        sparse.pq_eotf_nits,
        sparse.nonlinear_bt2020_to_scrgb,
        sparse._model_preview,
        sparse._frame_stats,
        sparse._materialize_capture_frame,
    )
    return artifact_cache.canonical_sha256({
        "functions": {
            function.__name__: inspect.getsource(function)
            for function in functions
        },
        "rec709_luma": sparse.REC709_LUMA.tolist(),
    })


def code_identity():
    """Hash only implementation that can change raw converted frame bytes."""
    value = artifact_cache.code_identities(code_paths())
    value["native_color_functions"] = _materializer_source_identity()
    return value


def verify_code_identity(expected):
    if code_identity() != expected:
        raise RuntimeError(
            "native-HDR raw preprocessing implementation changed during generation"
        )


def full_decode_command(ffmpeg: Path, source: Path, filter_text: str):
    """Return the exact full-cadence native-PQ raw decode command."""
    return [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-nostdin",
        "-threads", "1", "-i", str(source), "-map", "0:v:0", "-an",
        "-sn", "-dn", "-vf", filter_text, "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1",
    ]


def timing_payload(timing):
    fields = (
        "frame_count", "source_frame_rate", "nominal_frame_rate", "time_base",
        "stream_start_time_seconds", "stream_duration_seconds",
        "constant_frame_rate", "unique_frame_duration_ticks",
        "timestamp_source", "frames",
    )
    return {field: timing.get(field) for field in fields}


def cut_luma(preview_u8, width, height):
    image = Image.fromarray(preview_u8, "RGB").convert("L").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def cut_score(previous_luma, current_luma):
    if previous_luma.shape != current_luma.shape:
        raise RuntimeError("source cut-analysis geometry changed")
    return float(np.mean(
        np.abs(current_luma - previous_luma), dtype=np.float32
    ))


def decode_frames(*, ffmpeg, source, filter_text, source_width,
                  source_height, width, height, expected_frame_count,
                  timing, staging, video_id, cut_analysis_width,
                  cut_analysis_height):
    """Decode and color-convert the immutable native-HDR frame payload.

    Keeping this loop beside the raw cache contract means its file hash—not
    the much larger publication/label preparer—controls raw cache invalidation.
    """
    staging = Path(staging)
    model_root = staging / native_hdr_capture.MODEL_SOURCE_DIRECTORY
    model_root.mkdir(parents=True)
    frame_bytes = source_width * source_height * 3 * 4
    process = subprocess.Popen(
        full_decode_command(ffmpeg, source, filter_text),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("cannot open FFmpeg native-HDR decode pipes")
        records = []
        cut_rows = []
        previous_luma = None
        while True:
            chunks = bytearray()
            while len(chunks) < frame_bytes:
                chunk = process.stdout.read(frame_bytes - len(chunks))
                if not chunk:
                    break
                chunks.extend(chunk)
            if not chunks:
                break
            frame_id = len(records)
            if len(chunks) != frame_bytes:
                raise RuntimeError(f"{video_id}: FFmpeg returned a truncated frame")
            if frame_id >= expected_frame_count:
                raise RuntimeError(f"{video_id}: FFmpeg returned extra frames")
            planar = np.frombuffer(chunks, dtype="<f4").reshape(
                3, source_height, source_width
            )
            try:
                rgba_f16, preview, preview_u8 = sparse._materialize_capture_frame(
                    planar, source_width, source_height, width, height
                )
            except RuntimeError as error:
                raise RuntimeError(f"{video_id}: {error}") from error
            suffix = f"{frame_id:05d}"
            model_path = model_root / f"frame_{suffix}.scrgb16"
            rgba_f16.tofile(model_path)
            model_stat = model_path.stat()
            preview_path = staging / f"frame_{suffix}.png"
            Image.fromarray(preview_u8, "RGB").save(
                preview_path, format="PNG", compress_level=6, optimize=False
            )
            timing_row = timing["frames"][frame_id]
            records.append({
                "frame": frame_id,
                "path": (
                    f"{native_hdr_capture.MODEL_SOURCE_DIRECTORY}/"
                    f"frame_{suffix}.scrgb16"
                ),
                "size": model_stat.st_size,
                "mtime_ns": model_stat.st_mtime_ns,
                "sha256": sparse.sha256(model_path),
                "preview": f"frame_{suffix}.png",
                "preview_sha256": sparse.sha256(preview_path),
                "timestamp_seconds": timing_row["timestamp_seconds"],
                "stats": sparse._frame_stats(rgba_f16, preview),
            })
            luma = cut_luma(
                preview_u8, cut_analysis_width, cut_analysis_height
            )
            score = (None if previous_luma is None else
                     cut_score(previous_luma, luma))
            cut_rows.append({
                "frame": frame_id,
                "timestamp_ticks": timing_row["timestamp_ticks"],
                "timestamp_seconds": timing_row["timestamp_seconds"],
                "scene_start": frame_id == 0,
                "preview_mean_absolute_delta": score,
            })
            previous_luma = luma

        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(
                f"{video_id}: FFmpeg decode failed ({return_code}): "
                f"{stderr[-1000:]}"
            )
        if len(records) != expected_frame_count:
            raise RuntimeError(
                f"{video_id}: FFmpeg decoded {len(records)} frames; "
                f"expected {expected_frame_count}"
            )
        return records, cut_rows
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def cut_analysis_contract(width, height):
    return {
        "width": width,
        "height": height,
        "input": "production-model-preview-u8-rgb",
        "luma": "Pillow-RGB-to-L-then-bilinear-resize",
        "normalization": "uint8-divided-by-255-as-float32",
        "score": "float32-mean-absolute-delta",
    }


def identity(*, row, conversion_hash, width, height,
             cut_analysis_width, cut_analysis_height, code_identity,
             runtime_identity_value=None):
    timing = timing_payload(row["timing"])
    return artifact_cache.cache_identity(
        artifact_kind="apollo-chug-native-pq-raw-frames-v1",
        source={
            "bytes": row["download"]["bytes"],
            "sha256": row["download"]["sha256"],
        },
        selection={
            "split": row["split"],
            "source_frame_count": row["source_frame_count"],
            "timing_sha256": native_hdr_capture.canonical_sha256(timing),
        },
        preprocessing={
            "contract": CONTRACT,
            "width": width,
            "height": height,
            "source_width": int(row["audit"]["width"]),
            "source_height": int(row["audit"]["height"]),
            "conversion_contract_sha256": conversion_hash,
            "cut_analysis": cut_analysis_contract(
                cut_analysis_width, cut_analysis_height
            ),
            "native_runtime": (
                runtime_identity() if runtime_identity_value is None else
                runtime_identity_value
            ),
        },
        color_contract={"contract_sha256": conversion_hash},
        code=code_identity,
    )
