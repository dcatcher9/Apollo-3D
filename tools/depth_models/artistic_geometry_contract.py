"""Exact deployment-geometry contract for the artistic policy.

The global policy head does not observe the destination eye raster.  A policy can
therefore be deployed only on exact render tuples that participated in the
multi-geometry safe-frontier intersection used to build its labels.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct


GEOMETRY_SCHEMA = 1
GEOMETRY_CONTRACT = "exact-artistic-policy-render-tuples-v1"
COLOR_MODE_SDR = "sdr-srgb-8bit"
COLOR_MODE_HDR = "hdr-scrgb-fp16"
SUPPORTED_COLOR_MODES = {COLOR_MODE_SDR, COLOR_MODE_HDR}
PATCH = 14
DEFAULT_DEPTH_SHORT_SIDE = 432
DEFAULT_DEPTH_MAX_ASPECT = 4.0
MAX_WIDTH = 1008
MAX_HEIGHT = 1008
GEOMETRY_KEYS = (
    "source_width", "source_height",
    "model_input_width", "model_input_height",
    "depth_short_side", "depth_max_aspect",
    "eye_width", "eye_height",
    "content_scale_x", "content_scale_y",
    "disparity_raster_width", "disparity_raster_height",
    "color_mode",
)


def _round_patch(value):
    return max(PATCH, int(math.floor(value / PATCH + 0.5)) * PATCH)


def aspect_aligned_dims(width, height,
                        depth_short_side=DEFAULT_DEPTH_SHORT_SIDE,
                        depth_max_aspect=DEFAULT_DEPTH_MAX_ASPECT,
                        max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Mirror production DA-V2 input-shape selection exactly."""
    raw_aspect = _f32(_f32(width) / _f32(height))
    depth_max_aspect = _f32(max(1.0, float(depth_max_aspect)))
    if raw_aspect >= 1.0:
        aspect = _f32(min(raw_aspect, depth_max_aspect))
    else:
        aspect = _f32(1.0 / min(_f32(1.0 / raw_aspect), depth_max_aspect))
    max_width = max(PATCH, max_width // PATCH * PATCH)
    max_height = max(PATCH, max_height // PATCH * PATCH)
    requested = _round_patch(max(196, int(depth_short_side)))
    if aspect >= 1.0:
        for output_h in range(min(requested, max_height), PATCH - 1, -PATCH):
            output_w = _round_patch(output_h * aspect)
            if output_w <= max_width:
                return output_w, output_h
    else:
        for output_w in range(min(requested, max_width), PATCH - 1, -PATCH):
            output_h = _round_patch(output_w / aspect)
            if output_h <= max_height:
                return output_w, output_h
    return PATCH, PATCH


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def source_content_scales(source_width, source_height, eye_width, eye_height):
    """Mirror the float32 HLSL/evaluator aspect-fit operation order."""
    source_aspect = _f32(_f32(source_width) / _f32(source_height))
    eye_aspect = _f32(_f32(eye_width) / _f32(eye_height))
    if source_aspect > eye_aspect:
        return 1.0, _f32(eye_aspect / source_aspect)
    return _f32(source_aspect / eye_aspect), 1.0


def geometry_tuple(row, color_mode=COLOR_MODE_SDR,
                   depth_short_side=DEFAULT_DEPTH_SHORT_SIDE,
                   depth_max_aspect=DEFAULT_DEPTH_MAX_ASPECT):
    """Build one normalized exact tuple from a schema-8 selector row."""
    source_width = int(row["source_width"])
    source_height = int(row["source_height"])
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("deployment geometry has invalid source dimensions")
    model_width, model_height = aspect_aligned_dims(
        source_width, source_height,
        depth_short_side=depth_short_side,
        depth_max_aspect=depth_max_aspect,
        max_width=min(MAX_WIDTH, source_width),
        max_height=min(MAX_HEIGHT, source_height),
    )
    result = {
        "source_width": source_width,
        "source_height": source_height,
        "model_input_width": model_width,
        "model_input_height": model_height,
        "depth_short_side": max(196, int(depth_short_side)),
        "depth_max_aspect": float(max(1.0, depth_max_aspect)),
        "eye_width": int(row["eye_width"]),
        "eye_height": int(row["eye_height"]),
        "content_scale_x": float(row["content_scale_x"]),
        "content_scale_y": float(row["content_scale_y"]),
        "disparity_raster_width": int(row["disparity_raster_width"]),
        "disparity_raster_height": int(row["disparity_raster_height"]),
        "color_mode": str(row.get("color_mode") or color_mode),
    }
    validate_geometry_tuple(result)
    return result


def validate_geometry_tuple(value):
    if not isinstance(value, dict) or set(value) != set(GEOMETRY_KEYS):
        raise RuntimeError("deployment geometry tuple has missing or unknown fields")
    for key in (
            "source_width", "source_height", "model_input_width",
            "model_input_height", "depth_short_side", "eye_width", "eye_height",
            "disparity_raster_width", "disparity_raster_height"):
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise RuntimeError(f"deployment geometry has invalid {key}")
    if value["depth_short_side"] < 196:
        raise RuntimeError("deployment geometry depth_short_side is below runtime minimum")
    max_aspect = value["depth_max_aspect"]
    if (not isinstance(max_aspect, (int, float)) or isinstance(max_aspect, bool) or
            not math.isfinite(float(max_aspect)) or float(max_aspect) < 1.0):
        raise RuntimeError("deployment geometry has invalid depth_max_aspect")
    expected_model = aspect_aligned_dims(
        value["source_width"], value["source_height"],
        depth_short_side=value["depth_short_side"],
        depth_max_aspect=value["depth_max_aspect"],
        max_width=min(MAX_WIDTH, value["source_width"]),
        max_height=min(MAX_HEIGHT, value["source_height"]),
    )
    if expected_model != (
            value["model_input_width"], value["model_input_height"]):
        raise RuntimeError("deployment geometry has stale model-input dimensions")
    if (value["disparity_raster_width"] != value["eye_width"] or
            value["disparity_raster_height"] != value["eye_height"]):
        raise RuntimeError("deployment disparity raster must equal the full eye raster")
    for key in ("content_scale_x", "content_scale_y"):
        item = value[key]
        if (not isinstance(item, (int, float)) or isinstance(item, bool) or
                not math.isfinite(float(item)) or not 0.0 < float(item) <= 1.0):
            raise RuntimeError(f"deployment geometry has invalid {key}")
    expected_scales = source_content_scales(
        value["source_width"], value["source_height"],
        value["eye_width"], value["eye_height"],
    )
    actual_scales = (
        float(value["content_scale_x"]), float(value["content_scale_y"])
    )
    if any(abs(actual - expected) > 1e-7
           for actual, expected in zip(actual_scales, expected_scales)):
        raise RuntimeError("deployment geometry has inconsistent content scales")
    if value["color_mode"] not in SUPPORTED_COLOR_MODES:
        raise RuntimeError(
            "deployment geometry color mode is not validated by the current label loader"
        )
    return value


def canonical_geometry_tuple(value):
    """Return the single typed JSON representation used for hashing and identity.

    JSON considers ``4`` and ``4.0`` different byte strings even though both describe the same
    geometry. Normalize integer dimensions and floating controls explicitly so runtime contracts,
    evaluator output, and promotion manifests compare numeric semantics rather than parser types.
    """
    validate_geometry_tuple(value)
    integer_keys = {
        "source_width", "source_height", "model_input_width", "model_input_height",
        "depth_short_side", "eye_width", "eye_height", "disparity_raster_width",
        "disparity_raster_height",
    }
    float_keys = {"depth_max_aspect", "content_scale_x", "content_scale_y"}
    return {
        key: (int(value[key]) if key in integer_keys else
              float(value[key]) if key in float_keys else str(value[key]))
        for key in GEOMETRY_KEYS
    }


def tuple_key(value):
    normalized = canonical_geometry_tuple(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def build_allowlist(tuples):
    unique = {
        tuple_key(value): canonical_geometry_tuple(value) for value in tuples
    }
    if not unique:
        raise RuntimeError("deployment geometry allow-list is empty")
    return {
        "schema": GEOMETRY_SCHEMA,
        "contract": GEOMETRY_CONTRACT,
        "tuples": [unique[key] for key in sorted(unique)],
    }


def validate_allowlist(value):
    if (not isinstance(value, dict) or value.get("schema") != GEOMETRY_SCHEMA or
            value.get("contract") != GEOMETRY_CONTRACT or
            set(value) != {"schema", "contract", "tuples"}):
        raise RuntimeError("invalid deployment geometry allow-list contract")
    canonical = build_allowlist(value.get("tuples", ()))
    if value != canonical:
        raise RuntimeError("deployment geometry allow-list is not canonical")
    return value


def allowlist_sha256(value):
    validate_allowlist(value)
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
