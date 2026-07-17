#!/usr/bin/env python3
"""Shared fail-closed contract for SBS harness result provenance."""

HARNESS_SCHEMA = 28

COLOR_MODE_SDR = "sdr-srgb-8bit"
COLOR_MODE_LINEAR_SDR = "linear-sdr-fp16"
COLOR_MODE_HDR = "hdr-scrgb-fp16"

METRIC_PREVIEW_SDR = "native-srgb-v1"
METRIC_PREVIEW_HDR = (
    "source-relative-srgb-from-scrgb-white-normalized-v1"
)
METRIC_PREVIEW_NATIVE_HDR = (
    "perceptual-srgb-from-native-scrgb-reinhard-v1"
)

HDR_SOURCE_SDR = "native-sdr"
HDR_SOURCE_SIMULATED = "sdr-in-windows-hdr"
HDR_SOURCE_NATIVE_PQ = "native-pq-in-windows-hdr"


def expected_metric_preview_encoding(color_mode, hdr_source_kind=None):
    """Return the one admitted metric-preview encoding for ``color_mode``."""
    if color_mode == COLOR_MODE_HDR:
        if hdr_source_kind == HDR_SOURCE_NATIVE_PQ:
            return METRIC_PREVIEW_NATIVE_HDR
        if hdr_source_kind not in {None, HDR_SOURCE_SIMULATED}:
            raise RuntimeError(
                f"unsupported HDR source kind: {hdr_source_kind!r}"
            )
        return METRIC_PREVIEW_HDR
    if color_mode in {COLOR_MODE_SDR, COLOR_MODE_LINEAR_SDR}:
        if hdr_source_kind not in {None, HDR_SOURCE_SDR}:
            raise RuntimeError(
                f"SDR color mode has HDR source kind: {hdr_source_kind!r}"
            )
        return METRIC_PREVIEW_SDR
    raise RuntimeError(f"unsupported harness color mode: {color_mode!r}")


def validate_metric_preview_encoding(
        color_mode, encoding, origin="harness", hdr_source_kind=None):
    """Require explicit preview provenance; absent/legacy values are stale."""
    expected = expected_metric_preview_encoding(color_mode, hdr_source_kind)
    if encoding != expected:
        raise RuntimeError(
            f"{origin}: metric preview encoding {encoding!r} does not match "
            f"{color_mode!r}; expected {expected!r}"
        )
    return encoding


def input_variant_hdr_source_kind(variant):
    """Map one authenticated depth-input variant to harness provenance."""
    kind = variant.get("kind") if isinstance(variant, dict) else None
    if kind == "sdr-rgb8":
        return HDR_SOURCE_SDR
    if kind == "simulated-sdr-in-windows-hdr":
        return HDR_SOURCE_SIMULATED
    if kind == "native-pq-in-windows-hdr":
        return HDR_SOURCE_NATIVE_PQ
    raise RuntimeError(f"unsupported depth input variant kind: {kind!r}")


def input_variant_metric_preview_encoding(variant):
    return expected_metric_preview_encoding(
        variant["color_mode"], input_variant_hdr_source_kind(variant)
    )
