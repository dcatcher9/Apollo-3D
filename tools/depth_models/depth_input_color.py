#!/usr/bin/env python3
"""Deterministic CPU mirror of Apollo's depth-model input color pipeline.

The production compute shader samples the capture texture before converting it
to model-referred sRGB.  In Windows HDR, an SDR source has already passed
through the compositor into an FP16 linear-scRGB texture.  Mirroring that FP16
storage boundary before bilinear resize is therefore part of the input
identity, not an implementation detail.

SDR-origin preprocessing accepts RGB uint8, while authenticated native-HDR
preprocessing accepts only finite RGB/RGBA FP16 linear scRGB.  Keeping these as
separate APIs makes the source encoding unambiguous and prevents an accidental
SDR quantization or transfer-function round trip.  NumPy and Torch backends
follow the same operation contracts; Torch is imported only when requested.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


COLOR_CONTRACT_PATH = Path(__file__).with_name(
    "depth_input_color_contract.json"
)
COLOR_CONTRACT_SCHEMA = 2
COLOR_CONTRACT_NAME = "apollo-depth-input-color-v2"

COLOR_MODE_SDR = "sdr-srgb-8bit"
COLOR_MODE_HDR = "hdr-scrgb-fp16"

INPUT_VARIANT_SCHEMA = 2
INPUT_VARIANT_CONTRACT = "apollo-depth-input-variant-v2"
INPUT_KIND_SDR = "sdr-rgb8"
INPUT_KIND_WINDOWS_HDR = "simulated-sdr-in-windows-hdr"
INPUT_KIND_NATIVE_PQ = "native-pq-in-windows-hdr"
SDR_SOURCE_ENCODING = "srgb-rec709-unorm8"
NATIVE_PQ_SOURCE_ENCODING = "pq-bt2020nc-limited-yuv420p10+"
HDR_CAPTURE_ENCODING = "linear-scrgb-rec709-float16"
RAW_WHITE_ANCHORS = (1000, 2500, 3750, 5000, 6000)

_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
_LUMINANCE = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
_VARIANT_KEYS = {
    "schema",
    "contract",
    "kind",
    "color_mode",
    "source_encoding",
    "capture_encoding",
    "windows_sdr_white_level_raw",
    "windows_sdr_white_nits",
    "scrgb_white_scale",
    "color_contract_sha256",
}


def _expected_color_contract():
    return {
        "schema": COLOR_CONTRACT_SCHEMA,
        "contract": COLOR_CONTRACT_NAME,
        "color_modes": {
            "sdr": COLOR_MODE_SDR,
            "hdr": COLOR_MODE_HDR,
        },
        "sdr_source": {
            "encoding": SDR_SOURCE_ENCODING,
            "rgb_channel_order": "rgb",
            "pipeline": [
                "unorm8-to-float32",
                "bilinear-resize",
                "saturate",
                "imagenet-normalize",
                "nchw",
            ],
        },
        "simulated_sdr_in_windows_hdr": {
            "capture_encoding": HDR_CAPTURE_ENCODING,
            "scrgb_reference_white_nits": 80.0,
            "windows_sdr_white_level": {
                "raw_anchors": list(RAW_WHITE_ANCHORS),
                "raw_units_per_80_nits": 1000,
                "white_nits_formula": "raw*80/1000",
                "scrgb_scale_formula": "raw/1000",
            },
            "srgb_eotf": {
                "knee": 0.04045,
                "low_scale": 12.92,
                "high_add": 0.055,
                "high_scale": 1.055,
                "high_exponent": 2.4,
            },
            "pipeline": [
                "unorm8-to-float32",
                "srgb-eotf",
                "multiply-scrgb-white-scale",
                "float16-store",
                "float32-bilinear-sample",
                "clamp-negative-rgb",
                "luminance-reinhard",
                "uniform-peak-normalize",
                "saturate",
                "srgb-oetf",
                "imagenet-normalize",
                "nchw",
            ],
        },
        "native_pq_in_windows_hdr": {
            "source_encoding": NATIVE_PQ_SOURCE_ENCODING,
            "source_contract": {
                "color_range": "limited",
                "color_primaries": "bt2020",
                "color_matrix": "bt2020-non-constant-luminance",
                "color_transfer": "smpte-st-2084",
                "minimum_component_bits": 10,
            },
            "capture_encoding": HDR_CAPTURE_ENCODING,
            "scrgb_reference_white_nits": 80.0,
            "pq_eotf": {
                "m1": 2610.0 / 16384.0,
                "m2": 2523.0 / 32.0,
                "c1": 3424.0 / 4096.0,
                "c2": 2413.0 / 128.0,
                "c3": 2392.0 / 128.0,
                "peak_nits": 10000.0,
                "scale_policy": "absolute-pq-nits-no-mastering-peak-rescale",
            },
            "linear_bt2020_to_linear_rec709_d65": [
                [1.660491, -0.587641, -0.072850],
                [-0.124550, 1.132900, -0.008349],
                [-0.018151, -0.100579, 1.118730],
            ],
            "source_to_capture_pipeline": [
                "limited-yuv-to-nonlinear-bt2020-rgb",
                "smpte-st-2084-eotf-to-absolute-nits",
                "linear-bt2020-to-linear-rec709-d65",
                "divide-by-80-nits",
                "preserve-negative-and-superwhite-rgb",
                "float16-store",
            ],
            "model_input_pipeline": [
                "validate-finite-float16-scrgb",
                "float32-bilinear-sample",
                "clamp-negative-rgb",
                "luminance-reinhard",
                "uniform-peak-normalize",
                "saturate",
                "srgb-oetf",
                "imagenet-normalize",
                "nchw",
            ],
            "input_channels": {
                "accepted": ["rgb", "rgba"],
                "alpha": "ignored",
            },
        },
        "resize": {
            "filter": "bilinear",
            "coordinate_transform": "half-pixel",
            "address_mode": "clamp",
            "uv": (
                "((output_x+0.5)/output_width,"
                "(output_y+0.5)/output_height)"
            ),
            "post_interpolation_quantization": "none",
        },
        "hdr_to_model_srgb": {
            "negative_rgb": "clamp-to-zero",
            "rec709_luminance": [0.2126, 0.7152, 0.0722],
            "reinhard": "rgb/(1+luminance)",
            "peak_normalization": "rgb/max(max(rgb),1)",
            "srgb_oetf": {
                "knee": 0.0031308,
                "low_scale": 12.92,
                "high_scale": 1.055,
                "high_exponent": 1.0 / 2.4,
                "high_add": -0.055,
            },
        },
        "imagenet_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def canonical_json_bytes(value):
    """Return the canonical UTF-8 representation used by provenance hashes."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_color_contract(value):
    """Fail closed if the data contract and CPU implementation disagree."""
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError):
        canonical = None
    if (not isinstance(value, dict) or
            canonical != canonical_json_bytes(_expected_color_contract())):
        raise RuntimeError("invalid or unsupported depth input color contract")
    return value


def load_color_contract(path=COLOR_CONTRACT_PATH):
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot load depth input color contract: {path}") from error
    return validate_color_contract(payload)


def color_contract_sha256(contract=None):
    if contract is None:
        contract = load_color_contract()
    validate_color_contract(contract)
    return canonical_sha256(contract)


def _base_variant(contract):
    return {
        "schema": INPUT_VARIANT_SCHEMA,
        "contract": INPUT_VARIANT_CONTRACT,
        "source_encoding": SDR_SOURCE_ENCODING,
        "color_contract_sha256": color_contract_sha256(contract),
    }


def sdr_input_variant(contract=None):
    """Build the canonical native-SDR input identity."""
    if contract is None:
        contract = load_color_contract()
    value = _base_variant(contract)
    value.update({
        "kind": INPUT_KIND_SDR,
        "color_mode": COLOR_MODE_SDR,
        "capture_encoding": SDR_SOURCE_ENCODING,
        "windows_sdr_white_level_raw": None,
        "windows_sdr_white_nits": None,
        "scrgb_white_scale": None,
    })
    return value


def _validate_raw_white_level(raw_white_level):
    if (not isinstance(raw_white_level, int) or
            isinstance(raw_white_level, bool) or
            raw_white_level not in RAW_WHITE_ANCHORS):
        anchors = ", ".join(str(value) for value in RAW_WHITE_ANCHORS)
        raise RuntimeError(
            f"Windows SDR white level must be a canonical raw anchor: {anchors}"
        )
    return raw_white_level


def windows_hdr_input_variant(raw_white_level, contract=None):
    """Build one SDR-in-Windows-HDR FP16 scRGB input identity."""
    raw_white_level = _validate_raw_white_level(raw_white_level)
    if contract is None:
        contract = load_color_contract()
    value = _base_variant(contract)
    value.update({
        "kind": INPUT_KIND_WINDOWS_HDR,
        "color_mode": COLOR_MODE_HDR,
        "capture_encoding": HDR_CAPTURE_ENCODING,
        "windows_sdr_white_level_raw": raw_white_level,
        "windows_sdr_white_nits": raw_white_level * 80.0 / 1000.0,
        "scrgb_white_scale": raw_white_level / 1000.0,
    })
    return value


def native_pq_input_variant(contract=None):
    """Build the canonical native-PQ-in-Windows-HDR input identity.

    This identity describes the FP16 linear-scRGB capture surface consumed by
    production.  The native-video derivative manifest remains responsible for
    authenticating the source decoder, chroma reconstruction, PQ EOTF, and
    BT.2020-to-scRGB conversion against the color contract.
    """
    if contract is None:
        contract = load_color_contract()
    value = _base_variant(contract)
    value.update({
        "kind": INPUT_KIND_NATIVE_PQ,
        "color_mode": COLOR_MODE_HDR,
        "source_encoding": NATIVE_PQ_SOURCE_ENCODING,
        "capture_encoding": HDR_CAPTURE_ENCODING,
        "windows_sdr_white_level_raw": None,
        "windows_sdr_white_nits": None,
        "scrgb_white_scale": None,
    })
    return value


def validate_input_variant(value, contract=None):
    """Authenticate a canonical SDR or simulated Windows-HDR variant."""
    if not isinstance(value, dict) or set(value) != _VARIANT_KEYS:
        raise RuntimeError("depth input variant has missing or unknown fields")
    if contract is None:
        contract = load_color_contract()
    if value.get("kind") == INPUT_KIND_SDR:
        expected = sdr_input_variant(contract)
    elif value.get("kind") == INPUT_KIND_WINDOWS_HDR:
        raw_white_level = value.get("windows_sdr_white_level_raw")
        expected = windows_hdr_input_variant(raw_white_level, contract)
    elif value.get("kind") == INPUT_KIND_NATIVE_PQ:
        expected = native_pq_input_variant(contract)
    else:
        raise RuntimeError("unsupported depth input variant kind")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError):
        canonical = None
    if canonical != canonical_json_bytes(expected):
        raise RuntimeError("depth input variant is not canonical")
    return value


def input_variant_sha256(value, contract=None):
    validate_input_variant(value, contract)
    return canonical_sha256(value)


def _validate_numpy_rgb8(rgb):
    if not isinstance(rgb, np.ndarray):
        raise TypeError("NumPy preprocessing requires a numpy.ndarray")
    if rgb.dtype != np.uint8:
        raise TypeError("depth input must be RGB uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError("depth input must have shape HxWx3")
    return rgb


def _validate_numpy_scrgb_f16(scrgb):
    if not isinstance(scrgb, np.ndarray):
        raise TypeError("NumPy preprocessing requires a numpy.ndarray")
    if scrgb.dtype != np.float16:
        raise TypeError("native HDR depth input must be FP16 scRGB")
    if (scrgb.ndim != 3 or scrgb.shape[2] not in (3, 4) or
            scrgb.shape[0] <= 0 or scrgb.shape[1] <= 0):
        raise ValueError("native HDR depth input must have shape HxWx3 or HxWx4")
    if not np.isfinite(scrgb).all():
        raise ValueError("native HDR depth input contains non-finite scRGB")
    return scrgb[..., :3]


def _validate_native_pq_variant(variant):
    variant = validate_input_variant(variant)
    if (variant["kind"] != INPUT_KIND_NATIVE_PQ or
            variant["color_mode"] != COLOR_MODE_HDR or
            variant["capture_encoding"] != HDR_CAPTURE_ENCODING):
        raise TypeError(
            "FP16 scRGB preprocessing requires the native PQ input variant"
        )
    return variant


def _validate_output_size(width, height):
    for value, name in ((width, "width"), (height, "height")):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"output {name} must be a positive integer")
    return width, height


def _axis_samples(input_size, output_size):
    # Match the HLSL operation order: UV is formed at the output pixel center,
    # then the normalized sampler coordinate is mapped onto source texels.
    output = np.arange(output_size, dtype=np.float32)
    uv = (output + np.float32(0.5)) / np.float32(output_size)
    source = uv * np.float32(input_size) - np.float32(0.5)
    lower = np.floor(source).astype(np.int64)
    fraction = source - lower.astype(np.float32)
    upper = lower + 1
    return (
        np.clip(lower, 0, input_size - 1),
        np.clip(upper, 0, input_size - 1),
        fraction.astype(np.float32, copy=False),
    )


def _bilinear_resize_numpy(rgb, width, height):
    y0, y1, fy = _axis_samples(rgb.shape[0], height)
    x0, x1, fx = _axis_samples(rgb.shape[1], width)
    top_left = rgb[y0[:, None], x0[None, :], :]
    top_right = rgb[y0[:, None], x1[None, :], :]
    bottom_left = rgb[y1[:, None], x0[None, :], :]
    bottom_right = rgb[y1[:, None], x1[None, :], :]
    wx = fx[None, :, None]
    wy = fy[:, None, None]
    top = top_left + (top_right - top_left) * wx
    bottom = bottom_left + (bottom_right - bottom_left) * wx
    return (top + (bottom - top) * wy).astype(np.float32, copy=False)


def _srgb_eotf_numpy(rgb):
    low = rgb / np.float32(12.92)
    high = np.power(
        (rgb + np.float32(0.055)) / np.float32(1.055),
        np.float32(2.4),
    )
    return np.where(rgb <= np.float32(0.04045), low, high).astype(
        np.float32, copy=False
    )


def _linear_to_srgb_numpy(rgb):
    rgb = np.clip(rgb, np.float32(0.0), np.float32(1.0))
    low = rgb * np.float32(12.92)
    high = (
        np.float32(1.055) * np.power(rgb, np.float32(1.0 / 2.4)) -
        np.float32(0.055)
    )
    return np.where(rgb <= np.float32(0.0031308), low, high).astype(
        np.float32, copy=False
    )


def _hdr_scrgb_to_model_srgb_numpy(rgb):
    rgb = np.maximum(rgb, np.float32(0.0))
    luminance = np.maximum(
        np.sum(rgb * _LUMINANCE, axis=2, keepdims=True, dtype=np.float32),
        np.float32(0.0),
    )
    rgb = rgb / (np.float32(1.0) + luminance)
    peak = np.max(rgb, axis=2, keepdims=True)
    rgb = rgb / np.maximum(peak, np.float32(1.0))
    return _linear_to_srgb_numpy(rgb)


def _normalize_numpy(rgb):
    normalized = (rgb - _MEAN) / _STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def preprocess_rgb8_to_nchw_numpy(rgb, width, height, variant):
    """Preprocess one HxWx3 NumPy RGB8 frame into contiguous float32 CHW."""
    rgb = _validate_numpy_rgb8(rgb)
    width, height = _validate_output_size(width, height)
    variant = validate_input_variant(variant)
    source = rgb.astype(np.float32) / np.float32(255.0)
    if variant["kind"] == INPUT_KIND_SDR:
        model_srgb = np.clip(
            _bilinear_resize_numpy(source, width, height),
            np.float32(0.0),
            np.float32(1.0),
        )
    elif variant["kind"] == INPUT_KIND_WINDOWS_HDR:
        linear = _srgb_eotf_numpy(source)
        # Windows capture stores compositor output in R16G16B16A16_FLOAT.  The
        # round trip is intentionally before interpolation.
        scrgb = (
            linear * np.float32(variant["scrgb_white_scale"])
        ).astype(np.float16).astype(np.float32)
        resized = _bilinear_resize_numpy(scrgb, width, height)
        model_srgb = _hdr_scrgb_to_model_srgb_numpy(resized)
    else:
        raise TypeError(
            "native PQ input requires preprocess_scrgb_f16_to_nchw"
        )
    return _normalize_numpy(model_srgb)


def preprocess_scrgb_f16_to_nchw_numpy(scrgb, width, height, variant):
    """Preprocess one native-HDR FP16 scRGB frame into float32 CHW.

    The frame has already crossed the same R16G16B16A16_FLOAT storage
    boundary as Windows capture.  Convert to float32 before bilinear sampling,
    matching Texture2D<float4>.SampleLevel, and then use the production HDR
    model-input mapping without another quantization or source-space decode.
    """
    scrgb = _validate_numpy_scrgb_f16(scrgb)
    width, height = _validate_output_size(width, height)
    _validate_native_pq_variant(variant)
    source = scrgb.astype(np.float32)
    resized = _bilinear_resize_numpy(source, width, height)
    model_srgb = _hdr_scrgb_to_model_srgb_numpy(resized)
    return _normalize_numpy(model_srgb)


def _import_torch():
    try:
        torch = importlib.import_module("torch")
        functional = importlib.import_module("torch.nn.functional")
    except (ImportError, OSError) as error:
        raise RuntimeError("Torch preprocessing requires PyTorch") from error
    return torch, functional


def _validate_torch_rgb8(rgb, torch):
    if not isinstance(rgb, torch.Tensor):
        raise TypeError("Torch preprocessing requires a torch.Tensor")
    if rgb.dtype != torch.uint8:
        raise TypeError("depth input must be RGB uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
        raise ValueError("depth input must have shape HxWx3")
    return rgb


def _validate_torch_scrgb_f16(scrgb, torch):
    if not isinstance(scrgb, torch.Tensor):
        raise TypeError("Torch preprocessing requires a torch.Tensor")
    if scrgb.dtype != torch.float16:
        raise TypeError("native HDR depth input must be FP16 scRGB")
    if (scrgb.ndim != 3 or scrgb.shape[2] not in (3, 4) or
            scrgb.shape[0] <= 0 or scrgb.shape[1] <= 0):
        raise ValueError("native HDR depth input must have shape HxWx3 or HxWx4")
    if not bool(torch.isfinite(scrgb).all()):
        raise ValueError("native HDR depth input contains non-finite scRGB")
    return scrgb[..., :3]


def _bilinear_resize_torch(rgb, width, height, functional):
    nchw = rgb.permute(2, 0, 1).unsqueeze(0)
    return functional.interpolate(
        nchw,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).permute(1, 2, 0)


def _srgb_eotf_torch(rgb, torch):
    low = rgb / 12.92
    high = torch.pow((rgb + 0.055) / 1.055, 2.4)
    return torch.where(rgb <= 0.04045, low, high)


def _linear_to_srgb_torch(rgb, torch):
    rgb = torch.clamp(rgb, 0.0, 1.0)
    low = rgb * 12.92
    high = 1.055 * torch.pow(rgb, 1.0 / 2.4) - 0.055
    return torch.where(rgb <= 0.0031308, low, high)


def _hdr_scrgb_to_model_srgb_torch(rgb, torch):
    rgb = torch.clamp_min(rgb, 0.0)
    coefficients = torch.tensor(
        (0.2126, 0.7152, 0.0722), dtype=torch.float32, device=rgb.device
    )
    luminance = torch.clamp_min(
        torch.sum(rgb * coefficients, dim=2, keepdim=True), 0.0
    )
    rgb = rgb / (1.0 + luminance)
    peak = torch.amax(rgb, dim=2, keepdim=True)
    rgb = rgb / torch.clamp_min(peak, 1.0)
    return _linear_to_srgb_torch(rgb, torch)


def preprocess_rgb8_to_nchw_torch(rgb, width, height, variant):
    """Preprocess one HxWx3 Torch RGB8 tensor on its existing device."""
    torch, functional = _import_torch()
    rgb = _validate_torch_rgb8(rgb, torch)
    width, height = _validate_output_size(width, height)
    variant = validate_input_variant(variant)
    source = rgb.to(dtype=torch.float32) / 255.0
    if variant["kind"] == INPUT_KIND_SDR:
        model_srgb = torch.clamp(
            _bilinear_resize_torch(source, width, height, functional), 0.0, 1.0
        )
    elif variant["kind"] == INPUT_KIND_WINDOWS_HDR:
        linear = _srgb_eotf_torch(source, torch)
        scrgb = (
            linear * variant["scrgb_white_scale"]
        ).to(dtype=torch.float16).to(dtype=torch.float32)
        resized = _bilinear_resize_torch(scrgb, width, height, functional)
        model_srgb = _hdr_scrgb_to_model_srgb_torch(resized, torch)
    else:
        raise TypeError(
            "native PQ input requires preprocess_scrgb_f16_to_nchw"
        )
    mean = torch.tensor(
        (0.485, 0.456, 0.406), dtype=torch.float32, device=rgb.device
    )
    std = torch.tensor(
        (0.229, 0.224, 0.225), dtype=torch.float32, device=rgb.device
    )
    normalized = (model_srgb - mean) / std
    return normalized.permute(2, 0, 1).contiguous()


def preprocess_scrgb_f16_to_nchw_torch(scrgb, width, height, variant):
    """Torch implementation of native-HDR FP16 scRGB preprocessing."""
    torch, functional = _import_torch()
    scrgb = _validate_torch_scrgb_f16(scrgb, torch)
    width, height = _validate_output_size(width, height)
    _validate_native_pq_variant(variant)
    source = scrgb.to(dtype=torch.float32)
    resized = _bilinear_resize_torch(source, width, height, functional)
    model_srgb = _hdr_scrgb_to_model_srgb_torch(resized, torch)
    mean = torch.tensor(
        (0.485, 0.456, 0.406), dtype=torch.float32, device=scrgb.device
    )
    std = torch.tensor(
        (0.229, 0.224, 0.225), dtype=torch.float32, device=scrgb.device
    )
    normalized = (model_srgb - mean) / std
    return normalized.permute(2, 0, 1).contiguous()


def preprocess_rgb8_to_nchw(rgb, width, height, variant):
    """Dispatch deterministic RGB8 preprocessing to NumPy or Torch."""
    if isinstance(rgb, np.ndarray):
        return preprocess_rgb8_to_nchw_numpy(
            rgb, width, height, variant
        )
    torch, _ = _import_torch()
    if isinstance(rgb, torch.Tensor):
        return preprocess_rgb8_to_nchw_torch(
            rgb, width, height, variant
        )
    raise TypeError("depth input must be a NumPy array or Torch tensor")


def preprocess_scrgb_f16_to_nchw(scrgb, width, height, variant):
    """Dispatch native-HDR FP16 scRGB preprocessing to NumPy or Torch."""
    if isinstance(scrgb, np.ndarray):
        return preprocess_scrgb_f16_to_nchw_numpy(
            scrgb, width, height, variant
        )
    torch, _ = _import_torch()
    if isinstance(scrgb, torch.Tensor):
        return preprocess_scrgb_f16_to_nchw_torch(
            scrgb, width, height, variant
        )
    raise TypeError("native HDR depth input must be a NumPy array or Torch tensor")


def copy_color_contract():
    """Return a mutable copy for manifest assembly without sharing state."""
    return copy.deepcopy(load_color_contract())
