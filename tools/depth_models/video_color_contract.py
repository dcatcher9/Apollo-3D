#!/usr/bin/env python3
"""Probe video color metadata before OpenCV decodes training RGB frames.

OpenCV exposes decoded BGR8 but not enough provenance to distinguish SDR from
PQ/HLG input.  Feeding an implicit HDR-to-8-bit conversion into an SDR-labelled
dataset can clip exactly the highlights and edges the artistic policy is meant
to protect, so automatic admission fails closed on HDR/wide-gamut/high-bit-depth
sources.  A reviewed SDR override is available only for ambiguous metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess


HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
WIDE_GAMUT_PRIMARIES = {"bt2020", "smpte432", "smpte431"}
WIDE_GAMUT_SPACES = {"bt2020nc", "bt2020c"}
UNKNOWN_VALUES = {None, "", "unknown", "unspecified", "reserved"}

# FFmpeg pixel-format names do not have one universal bit-depth spelling.  The
# planar families encode component depth near the suffix (``yuv420p10le``),
# while packed formats encode either component depth (``xyz12le``) or total
# storage width (``rgb48le``).  Keep the exceptional packed formats explicit so
# an RGB storage width is never mistaken for component depth.
_COMPONENT_DEPTH_FORMAT = re.compile(
    r"^(?:(?:yuva?|gbr(?:a)?|gray|bayer_[a-z0-9_]+).*|xyz)"
    r"(9|10|12|14|16|32)(?:le|be)$"
)
_SEMIPLANAR_DEPTH_FORMAT = re.compile(
    r"^p[024](10|12|16)(?:le|be)$"
)
_PACKED_COMPONENT_DEPTH_FORMAT = re.compile(
    r"^(?:x2|a2)(?:rgb|bgr)(10)(?:le|be)$|"
    r"^y2(10|12|16)(?:le|be)$"
)
_PACKED_STORAGE_DEPTHS = {
    "rgb48le": 16,
    "rgb48be": 16,
    "bgr48le": 16,
    "bgr48be": 16,
    "rgba64le": 16,
    "rgba64be": 16,
    "bgra64le": 16,
    "bgra64be": 16,
    "argb64le": 16,
    "argb64be": 16,
    "abgr64le": 16,
    "abgr64be": 16,
    "ayuv64le": 16,
    "ayuv64be": 16,
    "v210": 10,
    "v410": 10,
    "nv20le": 10,
    "nv20be": 10,
    "xv30le": 10,
    "xv30be": 10,
    "xv36le": 12,
    "xv36be": 12,
}


def _normalized(value):
    return value.strip().lower() if isinstance(value, str) else value


def _pixel_format_bit_depth(pixel_format):
    if not pixel_format:
        return 8
    packed = _PACKED_STORAGE_DEPTHS.get(pixel_format)
    if packed is not None:
        return packed
    match = _SEMIPLANAR_DEPTH_FORMAT.fullmatch(pixel_format)
    if match:
        return int(match.group(1))
    match = _COMPONENT_DEPTH_FORMAT.fullmatch(pixel_format)
    if match:
        return int(match.group(1))
    match = _PACKED_COMPONENT_DEPTH_FORMAT.fullmatch(pixel_format)
    if match:
        return int(next(value for value in match.groups() if value is not None))
    return 8


def pixel_bit_depth(stream):
    raw = stream.get("bits_per_raw_sample")
    raw_depth = 0
    if isinstance(raw, str) and raw.isdigit() and int(raw) > 0:
        raw_depth = int(raw)
    elif isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        raw_depth = raw
    pixel_format = _normalized(stream.get("pix_fmt")) or ""
    # Prefer the more conservative signal when container metadata and the pixel
    # format disagree.  Underestimating depth here silently admits a lossy
    # implicit conversion; overestimating merely requires explicit review.
    return max(raw_depth, _pixel_format_bit_depth(pixel_format))


def classify_sdr_stream(stream, input_color="auto"):
    if input_color not in {"auto", "sdr"}:
        raise RuntimeError(f"unsupported input-color policy: {input_color}")
    transfer = _normalized(stream.get("color_transfer"))
    primaries = _normalized(stream.get("color_primaries"))
    color_space = _normalized(stream.get("color_space"))
    bit_depth = pixel_bit_depth(stream)
    if transfer in HDR_TRANSFERS:
        raise RuntimeError(
            f"HDR transfer {transfer!r} requires an explicit versioned "
            "HDR-to-SDR conversion before dataset preparation"
        )
    if primaries in WIDE_GAMUT_PRIMARIES or color_space in WIDE_GAMUT_SPACES:
        raise RuntimeError(
            "wide-gamut input requires an explicit versioned SDR conversion "
            f"(primaries={primaries!r}, space={color_space!r})"
        )
    ambiguous = transfer in UNKNOWN_VALUES or primaries in UNKNOWN_VALUES
    if input_color == "auto" and bit_depth > 8:
        raise RuntimeError(
            f"{bit_depth}-bit video cannot be admitted through implicit BGR8 decoding; "
            "convert it explicitly or review it with --input-color sdr"
        )
    return {
        "input_color_policy": input_color,
        "admission": (
            "user-reviewed-sdr" if input_color == "sdr"
            else "probed-no-hdr-signals"
        ),
        "decoder_output": "opencv-bgr8",
        "dataset_color_contract": "decoded-sdr-bgr8",
        "source_pixel_format": stream.get("pix_fmt"),
        "source_bit_depth": bit_depth,
        "source_color_range": stream.get("color_range"),
        "source_color_space": stream.get("color_space"),
        "source_color_transfer": stream.get("color_transfer"),
        "source_color_primaries": stream.get("color_primaries"),
        "metadata_ambiguous": ambiguous,
    }


def resolve_ffprobe(ffprobe=None):
    candidate = Path(ffprobe) if ffprobe else None
    if candidate is None:
        located = shutil.which("ffprobe")
        candidate = Path(located) if located else None
    if candidate is None or not candidate.is_file():
        raise RuntimeError(
            "ffprobe is required to authenticate video color; pass --ffprobe"
        )
    return candidate.resolve()


def probe_sdr_input(video: Path, ffprobe=None, input_color="auto"):
    executable = resolve_ffprobe(ffprobe)
    command = [
        str(executable), "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,color_range,color_space,color_transfer,"
        "color_primaries,bits_per_raw_sample",
        "-of", "json", str(video),
    ]
    process = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    if process.returncode != 0:
        detail = process.stderr.strip()[-1000:]
        raise RuntimeError(f"ffprobe failed for {video}: {detail}")
    try:
        payload = json.loads(process.stdout)
    except ValueError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video}") from error
    streams = payload.get("streams", []) if isinstance(payload, dict) else []
    if len(streams) != 1 or not isinstance(streams[0], dict):
        raise RuntimeError(f"ffprobe did not identify exactly one primary video stream: {video}")
    result = classify_sdr_stream(streams[0], input_color)
    version = subprocess.run(
        [str(executable), "-version"], capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30,
    )
    if version.returncode != 0:
        raise RuntimeError(f"cannot query ffprobe version: {executable}")
    result.update({
        "ffprobe": str(executable),
        "ffprobe_version": version.stdout.splitlines()[0] if version.stdout else "",
    })
    return result
