#!/usr/bin/env python3
"""Generate native and HLSL declarations for the complete depth-coordinate-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT / "tools" / "sbsbench" / "contracts" /
    "depth-coordinate-v2-v1.json"
)
CPP_TARGET = ROOT / "src" / "generated" / "depth_coordinate_v2_contract.h"
HLSL_TARGET = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx" /
    "include" / "depth_coordinate_v2_contract.generated.hlsl"
)
HLSL_OCR_ASSERT_TARGET = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx" /
    "include" / "depth_coordinate_v2_ocr_assert.generated.hlsl"
)
HOST_SBS_SHADER_CACHE_HEADER = ROOT / "src" / "host_sbs_shader_cache.h"
LIMITER_GROUP_THREADS = 32
LIMITER_Q_FRACTION_BITS = 30
LIMITER_Q_SCALE = 1 << LIMITER_Q_FRACTION_BITS

EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "vector_width", "calibrated_defaults", "constant_buffer", "frame_stats",
    "shadow_state", "model_calibrations", "capture_provenance", "shader_implementation",
    "subtitle_ocr", "final_parallax",
}


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


EXPECTED_FIELD_KEYS = {"index", "name", "type", "gpu_encoding", "initial"}
EXPECTED_LAYOUT_FIELD_KEYS = {"index", "name", "type"}
EXPECTED_DEFAULT_NAMES = (
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "gain_per_pop",
    "reference_pop_strength",
    "direct_container_limit",
    "max_horizontal_slope",
    "max_vertical_shear",
    "vertical_majorant_share",
    "convergence_curve_default",
)
EXPECTED_DEFAULT_KEYS = set(EXPECTED_DEFAULT_NAMES)
EXPECTED_MODEL_CALIBRATION_KEYS = {
    "calibration_id", "depth_model", "depth_model_url", "onnx_sha256",
    "raw_coordinate_scale", "calibrated_input_shapes", "preprocess",
}
EXPECTED_PREPROCESS_KEYS = {
    "profile", "source_closure_schema", "source_file", "source_entrypoint",
    "source_target", "source_compile_flags", "source_macro_count",
    "source_closure_sha256", "model_input_schema", "dtype", "layout", "channels",
    "patch_multiple", "maximum_dimension", "imagenet_mean", "imagenet_std", "stage",
}
EXPECTED_SHADER_IMPLEMENTATION_KEYS = {
    "source_closure_schema", "source_compile_flags", "source_macro_count", "source_specs",
    "source_closure_sha256",
}
EXPECTED_SHADER_SPEC_KEYS = {
    "source_file", "source_entrypoint", "source_target",
}
CANONICAL_FINAL_PARALLAX = {
    "schema": 2,
    "authority": "complete-atomic-subtitle-conditioned-r32f-live-render-authority",
    "publication_policy": (
        "authenticated-infer-or-cpu-known-publication-or-authenticated-cadence-due-"
        "subtitle-publication-direct-render"),
    "reuse_policy": (
        "ordinary-reuse-holds-complete-depth-ocr-slr-final-tuple-byte-for-byte;"
        "authenticated-cadence-due-reuse-holds-depth-and-publishes-current-ocr-or-"
        "abstention-slr-final-tuple-against-retained-base"),
    "invalid_policy": "fail-closed-flat",
    "current_rgb_policy": "always-current-never-retained",
}
CANONICAL_SUBTITLE_OCR = {
    "schema": 13,
    "logical_model": "ppocrv6_tiny_det_modelopt_fp16",
    "asset_path": "models/ppocrv6_tiny_det_modelopt045_mixed_fp16_fp32io.onnx",
    "artifact_onnx_sha256": (
        "169a233ba0ff7cac27f8ec7dccb6a406e614b25b21fe6a5638c423bf2118bb44"),
    "source_url": (
        "https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx/resolve/"
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c/inference.onnx?download=true"),
    "source_onnx_sha256": (
        "193bab7a04fca699a6c82e6abb5b81bdb28177f0abd4062552b04908dafb19f8"),
    "conversion_tool": "nvidia-modelopt",
    "conversion_version": "0.45.0",
    "conversion_recipe": "nvidia-modelopt-autocast-fp16-keep-io-fp32-v1",
    "conversion_calibration_profile": "apollo-live8-bottom960x160-v1",
    "engine_recipe": (
        "trt-strong-modelopt045-fp16-iofp32-tf32-fixed960x160-level5-v2"),
    "preprocess_profile": "apollo-ppocrv6-bottom-6x1-bgr-imagenet-v1",
    "source_crop": "bottom-6:1",
    "input_tensor": {
        "name": "x",
        "dtype": "float32",
        "layout": "NCHW",
        "shape": [1, 3, 160, 960],
        "channels": ["B", "G", "R"],
        "imagenet_mean": [0.485, 0.456, 0.406],
        "imagenet_std": [0.229, 0.224, 0.225],
    },
    "output_tensor": {
        "name": "fetch_name_0",
        "dtype": "float32",
        "layout": "NCHW",
        "shape": [1, 1, 160, 960],
    },
    "field_policy": {
        "detector_active_probability_threshold": 0.2,
        "detector_min_mean_score": 0.4,
        "locator_max_width_numerator": 9,
        "locator_max_width_denominator": 10,
        "locator_min_width_cells": 48,
        "locator_min_height_cells": 6,
        "locator_min_aspect_numerator": 2,
        "locator_min_aspect_denominator": 1,
        "locator_provisional_min_vertical_overlap_numerator": 3,
        "locator_provisional_min_vertical_overlap_denominator": 4,
        "locator_provisional_max_height_ratio": 2,
        "locator_provisional_max_center_y_delta_shorter_height": 1,
        "locator_corner_edge_divisor": 32,
        "locator_corner_bottom_rows": 16,
        "locator_match_iou_threshold": 0.6,
        "locator_death_grace_observations": 6,
        "locator_target_horizontal_fallback_max_radius_steps": 2,
        "locator_target_horizontal_step_denominator": 16,
        "locator_target_max_row_iqr_binocular_source_pixels": 8.0,
        "locator_target_max_row_median_delta_binocular_source_pixels": 4.0,
        "locator_target_max_residual_binocular_source_pixels": 8.0,
        "locator_target_max_unreliable_holds": 2,
        "locator_target_deadband_binocular_source_pixels": 1.0,
        "locator_target_ema_alpha": 0.125,
        "locator_target_max_slew_binocular_source_pixels": 0.25,
        "ocr_safe_row_top": 24,
        "ocr_safe_row_bottom": 155,
        "source_crop_aspect_width": 6,
        "source_crop_aspect_height": 1,
        "text_join_gap_cells": 4,
        "ribbon_join_gap_cells": 12,
        "ribbon_structural_gap_min_cells": 3,
        "ribbon_min_structural_gaps": 3,
        "ribbon_min_width_numerator": 1,
        "ribbon_min_width_denominator": 2,
        "ribbon_bottom_tolerance_pixels": 2,
        "ribbon_bottom_tolerance_projection":
            "exact-ceil-detector-edge-through-bottom-crop-v1",
        "ribbon_cover_pad_limit": 8,
    },
    "ocr_record": {
        "schema": 3,
        "tag": 0x3852434F,
        "word_count": 208,
        "header_word_count": 16,
        "box_word_count": 8,
        "box_flag_ribbon": 1,
        "box_known_flags": 1,
        "raw_box_offset": 16,
        "raw_box_capacity": 16,
        "final_box_offset": 144,
        "final_box_capacity": 8,
    },
    "locator_state": {
        "schema": 13,
        "tag": 0x33314C53,
        "word_count": 80,
        "header_word_count": 32,
        "rectangle_capacity": 4,
        "owner_offset": 32,
        "pending_offset": 48,
        "current_offset": 64,
        "kind_word": 31,
        "owner_kind_shift": 0,
        "pending_kind_shift": 4,
        "current_kind_shift": 8,
        "kind_mask": 15,
        "provisional_current_flag": 1 << 4,
        "provisional_target_word": 29,
        "provisional_fade_word": 30,
    },
    "condition_params": {
        "schema": 3,
        "tag": 0x33504353,
        "word_count": 6,
    },
}
EXPECTED_CONSTANT_FIELD_NAMES = (
    "raw_coordinate_scale",
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "requested_gain",
    "max_horizontal_slope",
    "direct_container_limit",
    "convergence_curve_default",
)
EXPECTED_FRAME_STAT_FIELD_NAMES = (
    "mean",
    "population_std",
    "minimum",
    "maximum",
    "valid_count",
    "texel_count",
    "valid",
    "reserved",
)
EXPECTED_SHADOW_STATE_FIELD_NAMES = (
    "center",
    "inverse_scale",
    "convergence_curve",
    "container_scale",
    "calibration_revision",
    "frame_valid",
    "confirmed_cut_count",
    "contract_tag_bits",
    "camera_center_integrity_bits",
    "renderer_authorization_bits",
    "mapping_state_reserved_1",
    "mapping_state_reserved_2",
)
CONTRACT_TAG_SENTINEL = "contract_tag"
PREPROCESS_SHADER_ROOT = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx")
PREPROCESS_SHADER_SPECS = (
    ("rgb_to_nchw_cs.hlsl", "main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "content_main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "pad_main", "cs_5_0"),
)
PARALLAX_V2_SHADER_SPECS = (
    ("rgb_to_nchw_cs.hlsl", "main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "content_main", "cs_5_0"),
    ("rgb_to_nchw_cs.hlsl", "pad_main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "main", "cs_5_0"),
    ("buffer_to_tex_cs.hlsl", "pad_main", "cs_5_0"),
    ("depth_minmax_ema_cs.hlsl", "main", "cs_5_0"),
    ("depth_hist_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_evidence_cs.hlsl", "main", "cs_5_0"),
    ("depth_scene_cut_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_valid_history_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_state_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_map_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_ownership_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_preprocess_cs.hlsl", "main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "cells_main", "cs_5_0"),
    ("host_sbs_ocr_boxes_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "resolve_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_prepare_main", "cs_5_0"),
    ("host_sbs_subtitle_locator_cs.hlsl", "condition_main", "cs_5_0"),
)
PARALLAX_V2_LIVE_RENDERER_SHADER_SPECS = (
    ("sbs_reprojection_v2_live_ps.hlsl", "main_ps", "ps_5_0"),
    ("sbs_reprojection_vs.hlsl", "main_vs", "vs_5_0"),
)
PARALLAX_V2_P010_Y_SHADER_SPECS = (
    ("sbs_reprojection_v2_p010_y_ps.hlsl", "main_p010_y_ps", "ps_5_0"),
)
PARALLAX_V2_DIAGNOSTIC_SHADER_SPECS = (
    ("sbs_reprojection_v2_diagnostics_ps.hlsl", "mapping_ps", "ps_5_0"),
    ("sbs_reprojection_v2_diagnostics_ps.hlsl", "mask_ps", "ps_5_0"),
)
RENDERER_SOURCE_CLOSURE_PINS = (
    ("parallax_v2_live_renderer_source_closure_sha256",
     PARALLAX_V2_LIVE_RENDERER_SHADER_SPECS, "live renderer"),
    ("parallax_v2_p010_y_source_closure_sha256",
     PARALLAX_V2_P010_Y_SHADER_SPECS, "optional P010 luma renderer"),
    ("parallax_v2_diagnostic_source_closure_sha256",
     PARALLAX_V2_DIAGNOSTIC_SHADER_SPECS, "diagnostic renderer"),
)
SOURCE_CLOSURE_DOMAIN = b"apollo-host-sbs-source-closure-v2\n"
SHADER_COMPILE_FLAGS = 0x00008800
SOURCE_CLOSURE_SCHEMA = 2
ALGORITHM_TAG_SHADER_DIGEST_SENTINEL = (
    "shader-source-closure-sha256-authenticated-separately-v1")
ANY_INCLUDE = re.compile(rb"^\s*#\s*include\b")
QUOTED_INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"')


def _identifier(value: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        raise ValueError(f"cannot turn {value!r} into an identifier")
    return identifier


def _upper_identifier(value: str) -> str:
    return _identifier(value).upper()


def _append_length_prefixed(output: bytearray, value: bytes) -> None:
    output.extend(len(value).to_bytes(8, "little"))
    output.extend(value)


def shader_source_closure_sha256(
        root: Path = PREPROCESS_SHADER_ROOT,
        specs: tuple[tuple[str, str, str], ...] = PREPROCESS_SHADER_SPECS) -> str:
    """Hash root specs plus the exact reachable quoted-include source closure."""

    canonical_root = root.resolve(strict=True)
    sources: dict[str, bytes] = {}
    include_edges: dict[tuple[str, str], str] = {}
    visited: set[Path] = set()

    def collect(requested: Path) -> None:
        path = requested.resolve(strict=True)
        try:
            relative = path.relative_to(canonical_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"shader dependency escapes source root: {path}") from exc
        if path in visited:
            return
        visited.add(path)
        source = path.read_bytes()
        sources[relative] = source
        normalized_for_scan = source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        for line in normalized_for_scan.split(b"\n"):
            match = QUOTED_INCLUDE.search(line)
            if match is None:
                if ANY_INCLUDE.search(line) is not None:
                    raise ValueError(
                        f"shader uses an unauthenticated non-quoted include: {relative}")
                continue
            include = Path(match.group(1).decode("utf-8"))
            candidate = path.parent / include
            if not candidate.exists():
                candidate = canonical_root / include
            resolved = candidate.resolve(strict=True)
            try:
                child = resolved.relative_to(canonical_root).as_posix()
            except ValueError as exc:
                raise ValueError(f"shader dependency escapes source root: {resolved}") from exc
            edge = (relative, match.group(1).decode("utf-8"))
            previous = include_edges.setdefault(edge, child)
            if previous != child:
                raise ValueError(f"shader include edge is ambiguous: {edge!r}")
            collect(resolved)

    for filename, entrypoint, target in specs:
        if not filename or not entrypoint or not target:
            raise ValueError("shader source closure specs must be non-empty")
        collect(canonical_root / filename)

    canonical = bytearray(SOURCE_CLOSURE_DOMAIN)
    canonical.extend(b"C")
    canonical.extend(SHADER_COMPILE_FLAGS.to_bytes(8, "little"))
    _append_length_prefixed(canonical, b"macros:none")
    for spec in specs:
        canonical.extend(b"S")
        for value in spec:
            _append_length_prefixed(canonical, value.encode("utf-8"))
    for (parent, include), child in sorted(include_edges.items()):
        canonical.extend(b"I")
        _append_length_prefixed(canonical, parent.encode("utf-8"))
        _append_length_prefixed(canonical, include.encode("utf-8"))
        _append_length_prefixed(canonical, child.encode("utf-8"))
    for path in sorted(sources):
        canonical.extend(b"F")
        _append_length_prefixed(canonical, path.encode("utf-8"))
        normalized = sources[path].replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        _append_length_prefixed(canonical, normalized)
    return hashlib.sha256(canonical).hexdigest()


def validate_renderer_source_closure_pins(
        shader_root: Path = PREPROCESS_SHADER_ROOT,
        pin_header: Path = HOST_SBS_SHADER_CACHE_HEADER) -> dict[str, str]:
    """Require native renderer pins to match their exact reachable shader closures.

    Both renderer closures reach the generated Depth Coordinate V2 HLSL include. Running the
    contract generator therefore also proves that regenerating that include did not leave either
    independently authenticated renderer pin stale.
    """

    try:
        header = pin_header.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read renderer closure pin header: {pin_header}") from exc
    validated: dict[str, str] = {}
    for name, specs, label in RENDERER_SOURCE_CLOSURE_PINS:
        matches = re.findall(
            rf"\b{re.escape(name)}\s*=\s*\"([0-9a-f]{{64}})\";",
            header,
        )
        if len(matches) != 1:
            raise ValueError(f"{label} source closure pin must appear exactly once")
        expected = shader_source_closure_sha256(shader_root, specs)
        if matches[0] != expected:
            raise ValueError(
                f"{label} source closure pin is stale: expected {expected}, "
                f"found {matches[0]}")
        validated[name] = expected
    return validated


def canonical_bytes(contract: dict[str, Any]) -> bytes:
    """Return the complete normalized manifest used for its full provenance digest."""

    return json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def contract_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def contract_tag_semantic_bytes(contract: dict[str, Any]) -> bytes:
    """Canonical tag input without the independently authenticated shader-body digest.

    The generated HLSL is itself part of the shader source closure, so hashing the closure digest
    into the GPU tag would be self-referential.  Replace only that digest with a fixed domain
    sentinel.  All shader specs/compiler policy and every non-implementation contract semantic
    remain in the tag; the complete manifest digest plus shader closure digest authenticate the
    final source bodies without a circular fixed-point search.
    """

    tag_contract = dict(contract)
    implementation = dict(contract["shader_implementation"])
    implementation["source_closure_sha256"] = ALGORITHM_TAG_SHADER_DIGEST_SENTINEL
    tag_contract["shader_implementation"] = implementation
    return canonical_bytes(tag_contract)


def contract_tag_semantic_digest(contract: dict[str, Any]) -> str:
    return hashlib.sha256(contract_tag_semantic_bytes(contract)).hexdigest()


def tag_is_finite_normal(tag: int) -> bool:
    """Whether ``tag`` survives an exact uint-as-float-as-uint GPU round trip."""

    exponent = (tag >> 23) & 0xFF
    return 0 < exponent < 0xFF


def contract_tag(contract: dict[str, Any]) -> int:
    # One GPU word is available and transported through a float4 via asfloat/asuint. Reject
    # zero/subnormal and NaN/Inf encodings because drivers may flush or canonicalize those bits.
    digest = hashlib.sha256(contract_tag_semantic_bytes(contract)).digest()
    tag = int.from_bytes(digest[:4], "big")
    nonce = 0
    while not tag_is_finite_normal(tag):
        nonce += 1
        candidate = hashlib.sha256(
            digest + b"\0depth-coordinate-v2-tag\0" + nonce.to_bytes(4, "big")
        ).digest()
        tag = int.from_bytes(candidate[:4], "big")
    return tag


def validate_contract(
        contract: Any, *, verify_shader_source_closure: bool = True,
        verify_preprocess_source_closure: bool = True) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise ValueError("manifest root must be an object")
    if set(contract) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            "manifest keys must be exactly " + ", ".join(sorted(EXPECTED_TOP_LEVEL_KEYS)))
    schema = contract.get("schema")
    if not isinstance(schema, int) or isinstance(schema, bool) or schema < 1:
        raise ValueError("manifest schema must be a positive integer")
    if contract.get("vector_width") != 4:
        raise ValueError("depth-coordinate-v2 state vector width must be exactly four")

    provenance = contract.get("capture_provenance")
    if (not isinstance(provenance, dict) or
            set(provenance) != {"schema", "manifest_key", "binding"} or
            not isinstance(provenance.get("schema"), int) or
            isinstance(provenance.get("schema"), bool) or provenance["schema"] < 1 or
            any(not isinstance(provenance.get(name), str) or not provenance[name]
                for name in ("manifest_key", "binding"))):
        raise ValueError(
            "capture_provenance must contain a positive schema, manifest_key, and binding")

    if contract.get("subtitle_ocr") != CANONICAL_SUBTITLE_OCR:
        raise ValueError(
            "subtitle_ocr must exactly match the authenticated PP-OCRv6/OCR8/SLR13 contract")
    if contract.get("final_parallax") != CANONICAL_FINAL_PARALLAX:
        raise ValueError(
            "final_parallax must exactly match the authenticated direct-render contract")

    shader_implementation = contract.get("shader_implementation")
    if (not isinstance(shader_implementation, dict) or
            set(shader_implementation) != EXPECTED_SHADER_IMPLEMENTATION_KEYS):
        raise ValueError(
            "shader_implementation keys must be exactly " +
            ", ".join(sorted(EXPECTED_SHADER_IMPLEMENTATION_KEYS)))
    if (shader_implementation.get("source_closure_schema") != SOURCE_CLOSURE_SCHEMA or
            shader_implementation.get("source_compile_flags") != SHADER_COMPILE_FLAGS or
            shader_implementation.get("source_macro_count") != 0):
        raise ValueError("shader_implementation compiler/source identity is unsupported")
    source_specs = shader_implementation.get("source_specs")
    if not isinstance(source_specs, list):
        raise ValueError("shader_implementation.source_specs must be an ordered list")
    parsed_specs: list[tuple[str, str, str]] = []
    for index, spec in enumerate(source_specs):
        if not isinstance(spec, dict) or set(spec) != EXPECTED_SHADER_SPEC_KEYS:
            raise ValueError(
                f"shader_implementation.source_specs[{index}] has an unknown schema")
        values = tuple(spec[name] for name in (
            "source_file", "source_entrypoint", "source_target"))
        if any(not isinstance(value, str) or not value for value in values):
            raise ValueError(
                f"shader_implementation.source_specs[{index}] values must be non-empty strings")
        parsed_specs.append(values)  # type: ignore[arg-type]
    if tuple(parsed_specs) != PARALLAX_V2_SHADER_SPECS:
        raise ValueError("shader_implementation.source_specs is not the selected V2 shader set")
    shader_digest = shader_implementation.get("source_closure_sha256")
    if (not isinstance(shader_digest, str) or
            re.fullmatch(r"[0-9a-f]{64}", shader_digest) is None):
        raise ValueError("shader_implementation.source_closure_sha256 must be lowercase SHA-256")
    if (verify_shader_source_closure and
            shader_digest != shader_source_closure_sha256(
                specs=PARALLAX_V2_SHADER_SPECS)):
        raise ValueError(
            "shader_implementation.source_closure_sha256 is stale for the selected V2 shaders")

    calibrated_defaults = contract.get("calibrated_defaults")
    if not isinstance(calibrated_defaults, dict) or set(calibrated_defaults) != EXPECTED_DEFAULT_KEYS:
        raise ValueError(
            "calibrated_defaults keys must be exactly " +
            ", ".join(sorted(EXPECTED_DEFAULT_KEYS)))
    for name, value in calibrated_defaults.items():
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                not math.isfinite(float(value))):
            raise ValueError(f"calibrated default {name} must be finite")
        if name == "convergence_curve_default":
            if float(value) != 0.0:
                raise ValueError("convergence_curve_default must be exactly zero")
        elif float(value) <= 0.0:
            raise ValueError(f"calibrated default {name} must be positive")
    if calibrated_defaults["max_horizontal_slope"] >= 1.0:
        raise ValueError("max_horizontal_slope must be below one")
    for name in ("max_horizontal_slope", "max_vertical_shear"):
        scaled = float(calibrated_defaults[name]) * LIMITER_Q_SCALE
        if not scaled.is_integer() or scaled <= 0.0 or scaled > 0xFFFFFFFF:
            raise ValueError(f"{name} must have an exact unsigned Q30 numerator")
    limiter_q_limit = math.ceil(
        _float32(calibrated_defaults["direct_container_limit"]) * LIMITER_Q_SCALE)
    if limiter_q_limit <= 0 or 4 * limiter_q_limit > 0x7FFFFFFF:
        raise ValueError("direct_container_limit is unsafe for signed Q30 limiter arithmetic")
    majorant_share = _float32(calibrated_defaults["vertical_majorant_share"])
    minorant_share = _float32(_float32(1.0) - majorant_share)
    if majorant_share <= 0.0 or minorant_share <= 0.0:
        raise ValueError(
            "vertical_majorant_share and its complement must remain positive in float32")

    calibrations = contract.get("model_calibrations")
    if not isinstance(calibrations, list) or not calibrations:
        raise ValueError("model_calibrations must be a non-empty list")
    seen_ids: set[str] = set()
    seen_models: set[str] = set()
    seen_hashes: set[str] = set()
    for position, calibration in enumerate(calibrations):
        prefix = f"model_calibrations[{position}]"
        if not isinstance(calibration, dict) or set(calibration) != EXPECTED_MODEL_CALIBRATION_KEYS:
            raise ValueError(
                f"{prefix} keys must be exactly " +
                ", ".join(sorted(EXPECTED_MODEL_CALIBRATION_KEYS)))
        calibration_id = calibration.get("calibration_id")
        model = calibration.get("depth_model")
        url = calibration.get("depth_model_url")
        digest = calibration.get("onnx_sha256")
        if not isinstance(calibration_id, str) or not calibration_id:
            raise ValueError(f"{prefix}.calibration_id must be a non-empty string")
        if (not isinstance(model, str) or not model or _identifier(model) != model):
            raise ValueError(f"{prefix}.depth_model must be an identifier")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise ValueError(f"{prefix}.depth_model_url must be an HTTPS URL")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{prefix}.onnx_sha256 must be lowercase SHA-256")
        scale = calibration.get("raw_coordinate_scale")
        if (not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                not math.isfinite(float(scale)) or float(scale) <= 0.0):
            raise ValueError(f"{prefix}.raw_coordinate_scale must be finite and positive")
        if calibration_id in seen_ids or model in seen_models or digest in seen_hashes:
            raise ValueError("model calibration ids, model names, and ONNX hashes must be unique")
        seen_ids.add(calibration_id)
        seen_models.add(model)
        seen_hashes.add(digest)

        preprocess = calibration.get("preprocess")
        if not isinstance(preprocess, dict) or set(preprocess) != EXPECTED_PREPROCESS_KEYS:
            raise ValueError(
                f"{prefix}.preprocess keys must be exactly " +
                ", ".join(sorted(EXPECTED_PREPROCESS_KEYS)))
        for name in ("profile", "stage"):
            if not isinstance(preprocess.get(name), str) or not preprocess[name]:
                raise ValueError(f"{prefix}.preprocess.{name} must be a non-empty string")
        source_closure_sha256 = preprocess.get("source_closure_sha256")
        if (not isinstance(source_closure_sha256, str) or
                re.fullmatch(r"[0-9a-f]{64}", source_closure_sha256) is None):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 must be lowercase SHA-256")
        expected_source_identity = {
            "source_closure_schema": SOURCE_CLOSURE_SCHEMA,
            "source_file": PREPROCESS_SHADER_SPECS[0][0],
            "source_entrypoint": PREPROCESS_SHADER_SPECS[0][1],
            "source_target": PREPROCESS_SHADER_SPECS[0][2],
            "source_compile_flags": SHADER_COMPILE_FLAGS,
            "source_macro_count": 0,
        }
        if any(preprocess.get(name) != value
               for name, value in expected_source_identity.items()):
            raise ValueError(
                f"{prefix}.preprocess shader source identity is unsupported")
        if (verify_preprocess_source_closure and
                source_closure_sha256 != shader_source_closure_sha256()):
            raise ValueError(
                f"{prefix}.preprocess.source_closure_sha256 is stale for the runtime shader")
        if (preprocess.get("model_input_schema") != 1 or
                preprocess.get("dtype") != "float32-le" or
                preprocess.get("layout") != "NCHW" or
                preprocess.get("channels") != ["R", "G", "B"]):
            raise ValueError(f"{prefix}.preprocess has unsupported model-input semantics")
        patch = preprocess.get("patch_multiple")
        maximum = preprocess.get("maximum_dimension")
        if (not isinstance(patch, int) or isinstance(patch, bool) or patch <= 0 or
                not isinstance(maximum, int) or isinstance(maximum, bool) or
                maximum < patch or maximum % patch):
            raise ValueError(
                f"{prefix}.preprocess dimensions need a positive patch multiple and aligned max")
        shapes = calibration.get("calibrated_input_shapes")
        if not isinstance(shapes, list) or not shapes:
            raise ValueError(f"{prefix}.calibrated_input_shapes must be non-empty")
        seen_shapes: set[tuple[int, int]] = set()
        for shape_index, shape in enumerate(shapes):
            if (not isinstance(shape, dict) or set(shape) != {"width", "height"} or
                    not isinstance(shape.get("width"), int) or
                    isinstance(shape.get("width"), bool) or shape["width"] <= 0 or
                    not isinstance(shape.get("height"), int) or
                    isinstance(shape.get("height"), bool) or shape["height"] <= 0 or
                    shape["width"] > maximum or shape["height"] > maximum or
                    shape["width"] % patch or shape["height"] % patch):
                raise ValueError(
                    f"{prefix}.calibrated_input_shapes[{shape_index}] violates preprocessing")
            dimensions = (shape["width"], shape["height"])
            if dimensions in seen_shapes:
                raise ValueError(f"{prefix}.calibrated_input_shapes contains a duplicate")
            seen_shapes.add(dimensions)
        for name in ("imagenet_mean", "imagenet_std"):
            values = preprocess.get(name)
            if (not isinstance(values, list) or len(values) != 3 or
                    any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                        not math.isfinite(float(value)) for value in values) or
                    (name == "imagenet_std" and any(float(value) <= 0.0 for value in values))):
                raise ValueError(f"{prefix}.preprocess.{name} must contain three finite values")

    def validate_float_layout(
            name: str, fields: Any, expected_names: tuple[str, ...]) -> None:
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"{name} fields must be a non-empty list")
        if len(fields) % contract["vector_width"]:
            raise ValueError(f"{name} field count must contain complete float4 vectors")
        if any(not isinstance(field, dict) or set(field) != EXPECTED_LAYOUT_FIELD_KEYS
               for field in fields):
            raise ValueError(
                f"every {name} field must contain exactly " +
                ", ".join(sorted(EXPECTED_LAYOUT_FIELD_KEYS)))
        if [field["index"] for field in fields] != list(range(len(fields))):
            raise ValueError(f"{name} fields must be ordered by contiguous index")
        names = [field["name"] for field in fields]
        if any(not isinstance(value, str) or not value or _identifier(value) != value
               for value in names):
            raise ValueError(f"{name} field names must be C/HLSL identifiers")
        if len(set(names)) != len(names):
            raise ValueError(f"{name} field names must be unique")
        if tuple(names) != expected_names:
            raise ValueError(
                f"{name} physical field order must be exactly " +
                ", ".join(expected_names))
        if any(field["type"] != "float32" for field in fields):
            raise ValueError(f"{name} currently supports only float32 fields")

    constant_buffer = contract.get("constant_buffer")
    if (not isinstance(constant_buffer, dict) or
            set(constant_buffer) != {"name", "register", "fields"}):
        raise ValueError("constant_buffer must contain exactly name, register, and fields")
    if (_identifier(str(constant_buffer.get("name", ""))) != constant_buffer.get("name") or
            not re.fullmatch(r"b[0-9]+", str(constant_buffer.get("register", "")))):
        raise ValueError("constant_buffer requires an identifier name and bN register")
    validate_float_layout(
        "constant_buffer", constant_buffer.get("fields"), EXPECTED_CONSTANT_FIELD_NAMES)

    frame_stats = contract.get("frame_stats")
    if (not isinstance(frame_stats, dict) or set(frame_stats) != {"source", "fields"} or
            not isinstance(frame_stats.get("source"), str) or not frame_stats["source"]):
        raise ValueError("frame_stats must contain a non-empty source and fields")
    validate_float_layout(
        "frame_stats", frame_stats.get("fields"), EXPECTED_FRAME_STAT_FIELD_NAMES)

    shadow_state = contract.get("shadow_state")
    if (not isinstance(shadow_state, dict) or
            set(shadow_state) != {"source", "capture", "fields"}):
        raise ValueError("shadow_state must contain exactly source, capture, and fields")
    for key in ("source", "capture"):
        if not isinstance(shadow_state.get(key), str) or not shadow_state[key]:
            raise ValueError(f"shadow_state {key} must be a non-empty string")
    fields = shadow_state.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("manifest fields must be a non-empty list")
    if len(fields) % contract["vector_width"]:
        raise ValueError("manifest field count must contain complete float4 vectors")
    if any(not isinstance(field, dict) or set(field) != EXPECTED_FIELD_KEYS
           for field in fields):
        raise ValueError(
            "every field must contain exactly " + ", ".join(sorted(EXPECTED_FIELD_KEYS)))
    indices = [field["index"] for field in fields]
    if indices != list(range(len(fields))):
        raise ValueError("manifest fields must be ordered by contiguous index")
    names = [field["name"] for field in fields]
    if any(not isinstance(name, str) or not name or _identifier(name) != name
           for name in names):
        raise ValueError("field names must be non-empty C/HLSL identifiers")
    if len(set(names)) != len(names):
        raise ValueError("field names must be unique")
    if tuple(names) != EXPECTED_SHADOW_STATE_FIELD_NAMES:
        raise ValueError(
            "shadow_state physical field order must be exactly " +
            ", ".join(EXPECTED_SHADOW_STATE_FIELD_NAMES))

    tag_fields = []
    for field in fields:
        field_type = field["type"]
        encoding = field["gpu_encoding"]
        initial = field["initial"]
        if field_type == "float32":
            if encoding != "float":
                raise ValueError(f"float32 field {field['name']} must use float encoding")
            if (not isinstance(initial, (int, float)) or isinstance(initial, bool) or
                    not math.isfinite(float(initial))):
                raise ValueError(f"float32 field {field['name']} needs a finite initial value")
        elif field_type == "uint32":
            if encoding != "uint_bits":
                raise ValueError(f"uint32 field {field['name']} must use uint_bits encoding")
            if initial == CONTRACT_TAG_SENTINEL:
                tag_fields.append(field)
            elif (not isinstance(initial, int) or isinstance(initial, bool) or
                  not 0 <= initial <= 0xFFFFFFFF):
                raise ValueError(f"uint32 field {field['name']} needs a uint32 initial value")
        else:
            raise ValueError(f"unsupported field type for {field['name']}: {field_type!r}")
    if (len(tag_fields) != 1 or tag_fields[0]["name"] != "contract_tag_bits" or
            tag_fields[0]["index"] !=
            EXPECTED_SHADOW_STATE_FIELD_NAMES.index("contract_tag_bits")):
        raise ValueError(
            "contract_tag_bits must be the sole uint32 contract tag sentinel at its "
            "authenticated physical index")
    contract_tag(contract)
    return contract


def load_contract(
        path: Path = MANIFEST, *, verify_shader_source_closure: bool = True,
        verify_preprocess_source_closure: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return validate_contract(
            json.load(stream),
            verify_shader_source_closure=verify_shader_source_closure,
            verify_preprocess_source_closure=verify_preprocess_source_closure,
        )


def _float_literal(value: float | int) -> str:
    numeric = float(value)
    if numeric.is_integer():
        return f"{int(numeric)}.0f"
    return f"{numeric:.9g}f"


def _initial_word_cpp(field: dict[str, Any], tag: int) -> str:
    initial = field["initial"]
    if initial == CONTRACT_TAG_SENTINEL:
        return "contract_tag"
    if field["type"] == "uint32":
        return f"{int(initial)}u"
    return f"std::bit_cast<std::uint32_t>({_float_literal(initial)})"


def render_cpp(contract: dict[str, Any]) -> str:
    fields = contract["shadow_state"]["fields"]
    defaults = contract["calibrated_defaults"]
    constants = contract["constant_buffer"]["fields"]
    frame_stats = contract["frame_stats"]["fields"]
    tag = contract_tag(contract)
    digest = contract_digest(contract)
    tag_semantic_digest = contract_tag_semantic_digest(contract)
    calibrations = contract["model_calibrations"]
    shader_implementation = contract["shader_implementation"]
    subtitle_ocr = contract["subtitle_ocr"]
    final_parallax = contract["final_parallax"]
    ocr_input = subtitle_ocr["input_tensor"]
    ocr_output = subtitle_ocr["output_tensor"]
    field_policy = subtitle_ocr["field_policy"]
    ocr_record = subtitle_ocr["ocr_record"]
    locator_state = subtitle_ocr["locator_state"]
    condition_params = subtitle_ocr["condition_params"]
    calibrated_shapes = [
        (calibration["calibration_id"], shape["width"], shape["height"])
        for calibration in calibrations
        for shape in calibration["calibrated_input_shapes"]
    ]
    lines = [
        "// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.",
        "// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <bit>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string_view>",
        "#include <type_traits>",
        "",
        "namespace models::depth_coordinate_v2 {",
        f"  inline constexpr std::uint32_t contract_schema = {contract['schema']}u;",
        f"  inline constexpr std::uint32_t contract_tag = 0x{tag:08X}u;",
        f'  inline constexpr std::string_view contract_canonical_sha256 = "{digest}";',
        f'  inline constexpr std::string_view contract_tag_semantic_sha256 = "{tag_semantic_digest}";',
        f'  inline constexpr std::string_view shadow_state_source = "{contract["shadow_state"]["source"]}";',
        f'  inline constexpr std::string_view shadow_state_capture = "{contract["shadow_state"]["capture"]}";',
        f'  inline constexpr std::string_view frame_stats_source = "{contract["frame_stats"]["source"]}";',
        f'  inline constexpr std::string_view constant_buffer_name = "{contract["constant_buffer"]["name"]}";',
        f'  inline constexpr std::string_view constant_buffer_register = "{contract["constant_buffer"]["register"]}";',
        f"  inline constexpr std::uint32_t capture_provenance_schema = "
        f"{contract['capture_provenance']['schema']}u;",
        f"  inline constexpr std::string_view capture_provenance_manifest_key = "
        f"{json.dumps(contract['capture_provenance']['manifest_key'])};",
        f"  inline constexpr std::string_view capture_provenance_binding = "
        f"{json.dumps(contract['capture_provenance']['binding'])};",
        f"  inline constexpr std::uint32_t final_parallax_contract_schema = "
        f"{final_parallax['schema']}u;",
        f"  inline constexpr std::string_view final_parallax_authority = "
        f"{json.dumps(final_parallax['authority'])};",
        f"  inline constexpr std::string_view final_parallax_publication_policy = "
        f"{json.dumps(final_parallax['publication_policy'])};",
        f"  inline constexpr std::string_view final_parallax_reuse_policy = "
        f"{json.dumps(final_parallax['reuse_policy'])};",
        f"  inline constexpr std::string_view final_parallax_invalid_policy = "
        f"{json.dumps(final_parallax['invalid_policy'])};",
        f"  inline constexpr std::string_view final_parallax_current_rgb_policy = "
        f"{json.dumps(final_parallax['current_rgb_policy'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_contract_schema = "
        f"{subtitle_ocr['schema']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_model_name = "
        f"{json.dumps(subtitle_ocr['logical_model'])};",
        f"  inline constexpr std::string_view subtitle_ocr_asset_path = "
        f"{json.dumps(subtitle_ocr['asset_path'])};",
        f"  inline constexpr std::string_view subtitle_ocr_artifact_onnx_sha256 = "
        f"{json.dumps(subtitle_ocr['artifact_onnx_sha256'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_url = "
        f"{json.dumps(subtitle_ocr['source_url'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_onnx_sha256 = "
        f"{json.dumps(subtitle_ocr['source_onnx_sha256'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_tool = "
        f"{json.dumps(subtitle_ocr['conversion_tool'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_version = "
        f"{json.dumps(subtitle_ocr['conversion_version'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_recipe = "
        f"{json.dumps(subtitle_ocr['conversion_recipe'])};",
        f"  inline constexpr std::string_view subtitle_ocr_conversion_calibration_profile = "
        f"{json.dumps(subtitle_ocr['conversion_calibration_profile'])};",
        f"  inline constexpr std::string_view subtitle_ocr_engine_recipe = "
        f"{json.dumps(subtitle_ocr['engine_recipe'])};",
        f"  inline constexpr std::string_view subtitle_ocr_preprocess_profile = "
        f"{json.dumps(subtitle_ocr['preprocess_profile'])};",
        f"  inline constexpr std::string_view subtitle_ocr_source_crop = "
        f"{json.dumps(subtitle_ocr['source_crop'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_name = "
        f"{json.dumps(ocr_input['name'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_dtype = "
        f"{json.dumps(ocr_input['dtype'])};",
        f"  inline constexpr std::string_view subtitle_ocr_input_layout = "
        f"{json.dumps(ocr_input['layout'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_n = {ocr_input['shape'][0]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_c = {ocr_input['shape'][1]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_height = {ocr_input['shape'][2]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_input_width = {ocr_input['shape'][3]}u;",
        "  inline constexpr std::array<std::string_view, 3> subtitle_ocr_input_channels {{" +
        ", ".join(json.dumps(value) for value in ocr_input["channels"]) + "}};",
        # Keep the manifest's decimal values as binary64 for canonical JSON provenance. HLSL
        # receives explicit float literals below; using float here would promote rounded binary32
        # values back to double when nlohmann serializes the dump descriptor.
        "  inline constexpr std::array<double, 3> subtitle_ocr_imagenet_mean {{" +
        ", ".join(json.dumps(value) for value in ocr_input["imagenet_mean"]) + "}};",
        "  inline constexpr std::array<double, 3> subtitle_ocr_imagenet_std {{" +
        ", ".join(json.dumps(value) for value in ocr_input["imagenet_std"]) + "}};",
        f"  inline constexpr std::string_view subtitle_ocr_output_name = "
        f"{json.dumps(ocr_output['name'])};",
        f"  inline constexpr std::string_view subtitle_ocr_output_dtype = "
        f"{json.dumps(ocr_output['dtype'])};",
        f"  inline constexpr std::string_view subtitle_ocr_output_layout = "
        f"{json.dumps(ocr_output['layout'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_n = {ocr_output['shape'][0]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_c = {ocr_output['shape'][1]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_height = {ocr_output['shape'][2]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_output_width = {ocr_output['shape'][3]}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_schema = {ocr_record['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_tag = 0x{ocr_record['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_word_count = {ocr_record['word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_record_header_word_count = {ocr_record['header_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_word_count = {ocr_record['box_word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_flag_ribbon = {ocr_record['box_flag_ribbon']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_box_known_flags = {ocr_record['box_known_flags']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_raw_box_offset = {ocr_record['raw_box_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_raw_box_capacity = {ocr_record['raw_box_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_final_box_offset = {ocr_record['final_box_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_final_box_capacity = {ocr_record['final_box_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_safe_row_top = "
        f"{field_policy['ocr_safe_row_top']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_safe_row_bottom = "
        f"{field_policy['ocr_safe_row_bottom']}u;",
        f"  inline constexpr float subtitle_ocr_active_probability_threshold = "
        f"{_float_literal(field_policy['detector_active_probability_threshold'])};",
        f"  inline constexpr float subtitle_ocr_min_mean_score = "
        f"{_float_literal(field_policy['detector_min_mean_score'])};",
        f"  inline constexpr std::uint32_t subtitle_locator_max_width_numerator = "
        f"{field_policy['locator_max_width_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_max_width_denominator = "
        f"{field_policy['locator_max_width_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_min_width_cells = "
        f"{field_policy['locator_min_width_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_min_height_cells = "
        f"{field_policy['locator_min_height_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_min_aspect_numerator = "
        f"{field_policy['locator_min_aspect_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_min_aspect_denominator = "
        f"{field_policy['locator_min_aspect_denominator']}u;",
        "  inline constexpr std::uint32_t "
        "subtitle_locator_provisional_min_vertical_overlap_numerator = "
        f"{field_policy['locator_provisional_min_vertical_overlap_numerator']}u;",
        "  inline constexpr std::uint32_t "
        "subtitle_locator_provisional_min_vertical_overlap_denominator = "
        f"{field_policy['locator_provisional_min_vertical_overlap_denominator']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_provisional_max_height_ratio = "
        f"{field_policy['locator_provisional_max_height_ratio']}u;",
        "  inline constexpr std::uint32_t "
        "subtitle_locator_provisional_max_center_y_delta_shorter_height = "
        f"{field_policy['locator_provisional_max_center_y_delta_shorter_height']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_corner_edge_divisor = "
        f"{field_policy['locator_corner_edge_divisor']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_corner_bottom_rows = "
        f"{field_policy['locator_corner_bottom_rows']}u;",
        f"  inline constexpr float subtitle_locator_match_iou_threshold = "
        f"{_float_literal(field_policy['locator_match_iou_threshold'])};",
        f"  inline constexpr std::uint32_t subtitle_locator_death_grace_observations = "
        f"{field_policy['locator_death_grace_observations']}u;",
        f"  inline constexpr std::uint32_t subtitle_target_horizontal_fallback_max_radius_steps = "
        f"{field_policy['locator_target_horizontal_fallback_max_radius_steps']}u;",
        f"  inline constexpr std::uint32_t subtitle_target_horizontal_step_denominator = "
        f"{field_policy['locator_target_horizontal_step_denominator']}u;",
        f"  inline constexpr float subtitle_target_max_row_iqr_binocular_source_pixels = "
        f"{_float_literal(field_policy['locator_target_max_row_iqr_binocular_source_pixels'])};",
        f"  inline constexpr float subtitle_target_max_row_median_delta_binocular_source_pixels = "
        f"{_float_literal(field_policy['locator_target_max_row_median_delta_binocular_source_pixels'])};",
        f"  inline constexpr float subtitle_target_max_residual_binocular_source_pixels = "
        f"{_float_literal(field_policy['locator_target_max_residual_binocular_source_pixels'])};",
        f"  inline constexpr std::uint32_t subtitle_target_max_unreliable_holds = "
        f"{field_policy['locator_target_max_unreliable_holds']}u;",
        f"  inline constexpr float subtitle_target_deadband_binocular_source_pixels = "
        f"{_float_literal(field_policy['locator_target_deadband_binocular_source_pixels'])};",
        f"  inline constexpr float subtitle_target_ema_alpha = "
        f"{_float_literal(field_policy['locator_target_ema_alpha'])};",
        f"  inline constexpr float subtitle_target_max_slew_binocular_source_pixels = "
        f"{_float_literal(field_policy['locator_target_max_slew_binocular_source_pixels'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_crop_aspect_width = "
        f"{field_policy['source_crop_aspect_width']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_crop_aspect_height = "
        f"{field_policy['source_crop_aspect_height']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_text_join_gap_cells = "
        f"{field_policy['text_join_gap_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_join_gap_cells = "
        f"{field_policy['ribbon_join_gap_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_structural_gap_min_cells = "
        f"{field_policy['ribbon_structural_gap_min_cells']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_structural_gaps = "
        f"{field_policy['ribbon_min_structural_gaps']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_width_numerator = "
        f"{field_policy['ribbon_min_width_numerator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_min_width_denominator = "
        f"{field_policy['ribbon_min_width_denominator']}u;",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_bottom_tolerance_pixels = "
        f"{field_policy['ribbon_bottom_tolerance_pixels']}u;",
        f"  inline constexpr std::string_view subtitle_ocr_ribbon_bottom_tolerance_projection = "
        f"{json.dumps(field_policy['ribbon_bottom_tolerance_projection'])};",
        f"  inline constexpr std::uint32_t subtitle_ocr_ribbon_cover_pad_limit = "
        f"{field_policy['ribbon_cover_pad_limit']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_state_schema = {locator_state['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_state_tag = 0x{locator_state['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_state_word_count = {locator_state['word_count']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_header_word_count = {locator_state['header_word_count']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_rectangle_capacity = "
        f"{locator_state['rectangle_capacity']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_owner_offset = {locator_state['owner_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_pending_offset = {locator_state['pending_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_current_offset = {locator_state['current_offset']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_kind_word = {locator_state['kind_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_owner_kind_shift = {locator_state['owner_kind_shift']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_pending_kind_shift = "
        f"{locator_state['pending_kind_shift']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_current_kind_shift = "
        f"{locator_state['current_kind_shift']}u;",
        f"  inline constexpr std::uint32_t subtitle_locator_kind_mask = {locator_state['kind_mask']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_provisional_current_flag = "
        f"{locator_state['provisional_current_flag']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_provisional_target_word = "
        f"{locator_state['provisional_target_word']}u;",
        "  inline constexpr std::uint32_t subtitle_locator_provisional_fade_word = "
        f"{locator_state['provisional_fade_word']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_schema = "
        f"{condition_params['schema']}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_tag = "
        f"0x{condition_params['tag']:08X}u;",
        f"  inline constexpr std::uint32_t subtitle_condition_param_word_count = "
        f"{condition_params['word_count']}u;",
        "  static_assert(subtitle_ocr_input_n == 1u && subtitle_ocr_input_c == 3u);",
        "  static_assert(subtitle_ocr_output_n == 1u && subtitle_ocr_output_c == 1u);",
        "  static_assert(subtitle_ocr_active_probability_threshold > 0.0f &&",
        "                subtitle_ocr_active_probability_threshold < 1.0f);",
        "  static_assert(subtitle_ocr_min_mean_score >",
        "                subtitle_ocr_active_probability_threshold);",
        "  static_assert(subtitle_ocr_min_mean_score <= 1.0f);",
        "  static_assert(subtitle_locator_min_width_cells > 0u);",
        "  static_assert(subtitle_locator_min_height_cells > 0u);",
        "  static_assert(subtitle_locator_min_aspect_numerator > 0u);",
        "  static_assert(subtitle_locator_min_aspect_denominator > 0u);",
        "  static_assert(subtitle_locator_provisional_min_vertical_overlap_numerator > 0u);",
        "  static_assert(subtitle_locator_provisional_min_vertical_overlap_denominator >= "
        "                subtitle_locator_provisional_min_vertical_overlap_numerator);",
        "  static_assert(subtitle_locator_provisional_max_height_ratio >= 1u);",
        "  static_assert(subtitle_locator_provisional_max_center_y_delta_shorter_height >= 1u);",
        "  static_assert(subtitle_locator_corner_edge_divisor > 1u);",
        "  static_assert(subtitle_locator_corner_bottom_rows > 0u);",
        "  static_assert(subtitle_locator_match_iou_threshold > 0.0f &&",
        "                subtitle_locator_match_iou_threshold <= 1.0f);",
        "  static_assert(subtitle_locator_death_grace_observations > 0u);",
        "  static_assert(subtitle_target_max_row_iqr_binocular_source_pixels > 0.0f);",
        "  static_assert(subtitle_target_max_row_median_delta_binocular_source_pixels > 0.0f);",
        "  static_assert(subtitle_target_max_residual_binocular_source_pixels > 0.0f);",
        "  static_assert(subtitle_target_max_unreliable_holds > 0u);",
        "  static_assert(subtitle_target_deadband_binocular_source_pixels > 0.0f);",
        "  static_assert(subtitle_target_ema_alpha > 0.0f &&",
        "                subtitle_target_ema_alpha < 1.0f);",
        "  static_assert(subtitle_target_max_slew_binocular_source_pixels > 0.0f &&",
        "                subtitle_target_max_slew_binocular_source_pixels <=",
        "                  subtitle_target_deadband_binocular_source_pixels);",
        "  static_assert(subtitle_ocr_text_join_gap_cells > 0u);",
        "  static_assert(subtitle_ocr_text_join_gap_cells <",
        "                subtitle_ocr_ribbon_join_gap_cells);",
        "  static_assert(subtitle_ocr_ribbon_join_gap_cells >=",
        "                subtitle_ocr_ribbon_structural_gap_min_cells);",
        "  static_assert(subtitle_ocr_ribbon_bottom_tolerance_pixels <=",
        "                subtitle_ocr_safe_row_bottom);",
        "  static_assert(subtitle_ocr_final_box_offset == subtitle_ocr_raw_box_offset +",
        "                subtitle_ocr_raw_box_capacity * subtitle_ocr_box_word_count);",
        "  static_assert(subtitle_ocr_record_word_count == subtitle_ocr_final_box_offset +",
        "                subtitle_ocr_final_box_capacity * subtitle_ocr_box_word_count);",
        "  static_assert(subtitle_locator_pending_offset == subtitle_locator_owner_offset +",
        "                subtitle_locator_rectangle_capacity * 4u);",
        "  static_assert(subtitle_locator_current_offset == subtitle_locator_pending_offset +",
        "                subtitle_locator_rectangle_capacity * 4u);",
        "  static_assert(subtitle_locator_state_word_count == subtitle_locator_current_offset +",
        "                subtitle_locator_rectangle_capacity * 4u);",
        "  static_assert(subtitle_target_horizontal_fallback_max_radius_steps == 2u);",
        "  static_assert(subtitle_target_horizontal_step_denominator == 16u);",
        "  static_assert(subtitle_condition_param_word_count == 6u);",
        f"  inline constexpr std::uint32_t shader_source_closure_schema = "
        f"{shader_implementation['source_closure_schema']}u;",
        f"  inline constexpr std::uint32_t shader_source_compile_flags = "
        f"{shader_implementation['source_compile_flags']}u;",
        f"  inline constexpr std::uint32_t shader_source_macro_count = "
        f"{shader_implementation['source_macro_count']}u;",
        f"  inline constexpr std::string_view shader_source_closure_sha256 = "
        f"{json.dumps(shader_implementation['source_closure_sha256'])};",
        "",
    ]
    lines.extend([
        "  struct shader_source_spec_t {",
        "    std::string_view source_file;",
        "    std::string_view source_entrypoint;",
        "    std::string_view source_target;",
        "  };",
        "",
        f"  inline constexpr std::array<shader_source_spec_t, "
        f"{len(shader_implementation['source_specs'])}> shader_source_specs {{{{",
    ])
    lines.extend(
        "    {%s, %s, %s}," % (
            json.dumps(spec["source_file"]),
            json.dumps(spec["source_entrypoint"]),
            json.dumps(spec["source_target"]),
        )
        for spec in shader_implementation["source_specs"]
    )
    lines.extend([
        "  }};",
        "",
    ])
    lines.extend(
        f"  inline constexpr float {_identifier(name)} = {_float_literal(defaults[name])};"
        for name in EXPECTED_DEFAULT_NAMES
    )
    lines.extend([
        f"  inline constexpr std::uint32_t limiter_group_threads = {LIMITER_GROUP_THREADS}u;",
        f"  inline constexpr std::uint32_t limiter_q_fraction_bits = "
        f"{LIMITER_Q_FRACTION_BITS}u;",
        f"  inline constexpr std::uint32_t limiter_q_scale = {LIMITER_Q_SCALE}u;",
        f"  inline constexpr std::int32_t limiter_container_q_limit = "
        f"{math.ceil(_float32(defaults['direct_container_limit']) * LIMITER_Q_SCALE)};",
        f"  inline constexpr std::uint32_t limiter_horizontal_step_q_numerator = "
        f"{int(float(defaults['max_horizontal_slope']) * LIMITER_Q_SCALE)}u;",
        f"  inline constexpr std::uint32_t limiter_vertical_step_q_numerator = "
        f"{int(float(defaults['max_vertical_shear']) * LIMITER_Q_SCALE)}u;",
        "  inline constexpr std::string_view direct_parallax_decode_expression = "
        f'"(encoded * 2 - 1) * {json.dumps(float(defaults["direct_container_limit"]))}";',
        "  static_assert(convergence_curve_default == 0.0f);",
        "  static_assert(max_horizontal_slope > 0.0f && max_horizontal_slope < 1.0f);",
        "  static_assert(max_vertical_shear > 0.0f);",
        "  static_assert(vertical_majorant_share > 0.0f && vertical_majorant_share < 1.0f);",
        "",
        "  struct model_preprocess_contract_t {",
        "    std::string_view profile;",
        "    std::uint32_t source_closure_schema;",
        "    std::string_view source_file;",
        "    std::string_view source_entrypoint;",
        "    std::string_view source_target;",
        "    std::uint32_t source_compile_flags;",
        "    std::uint32_t source_macro_count;",
        "    std::string_view source_closure_sha256;",
        "    std::uint32_t model_input_schema;",
        "    std::string_view dtype;",
        "    std::string_view layout;",
        "    std::array<std::string_view, 3> channels;",
        "    std::uint32_t patch_multiple;",
        "    std::uint32_t maximum_dimension;",
        "    std::array<float, 3> imagenet_mean;",
        "    std::array<float, 3> imagenet_std;",
        "    std::string_view stage;",
        "  };",
        "",
        "  struct model_calibration_t {",
        "    std::string_view calibration_id;",
        "    std::string_view depth_model;",
        "    std::string_view depth_model_url;",
        "    std::string_view onnx_sha256;",
        "    float raw_coordinate_scale;",
        "    model_preprocess_contract_t preprocess;",
        "  };",
        "",
        "  struct model_calibrated_shape_t {",
        "    std::string_view calibration_id;",
        "    std::uint32_t width;",
        "    std::uint32_t height;",
        "  };",
        "",
        f"  inline constexpr std::array<model_calibration_t, {len(calibrations)}> "
        "model_calibrations {{",
    ])
    for calibration in calibrations:
        preprocess = calibration["preprocess"]
        means = ", ".join(_float_literal(value) for value in preprocess["imagenet_mean"])
        stds = ", ".join(_float_literal(value) for value in preprocess["imagenet_std"])
        channels = ", ".join(json.dumps(value) for value in preprocess["channels"])
        lines.extend([
            "    {",
            f"      {json.dumps(calibration['calibration_id'])},",
            f"      {json.dumps(calibration['depth_model'])},",
            f"      {json.dumps(calibration['depth_model_url'])},",
            f"      {json.dumps(calibration['onnx_sha256'])},",
            f"      {_float_literal(calibration['raw_coordinate_scale'])},",
            "      {",
            f"        {json.dumps(preprocess['profile'])},",
            f"        {preprocess['source_closure_schema']}u,",
            f"        {json.dumps(preprocess['source_file'])},",
            f"        {json.dumps(preprocess['source_entrypoint'])},",
            f"        {json.dumps(preprocess['source_target'])},",
            f"        {preprocess['source_compile_flags']}u,",
            f"        {preprocess['source_macro_count']}u,",
            f"        {json.dumps(preprocess['source_closure_sha256'])},",
            f"        {preprocess['model_input_schema']}u,",
            f"        {json.dumps(preprocess['dtype'])},",
            f"        {json.dumps(preprocess['layout'])},",
            f"        {{{{{channels}}}}},",
            f"        {preprocess['patch_multiple']}u,",
            f"        {preprocess['maximum_dimension']}u,",
            f"        {{{{{means}}}}},",
            f"        {{{{{stds}}}}},",
            f"        {json.dumps(preprocess['stage'])},",
            "      },",
            "    },",
        ])
    lines.extend([
        "  }};",
        "",
        f"  inline constexpr std::array<model_calibrated_shape_t, {len(calibrated_shapes)}> "
        "model_calibrated_shapes {{",
    ])
    lines.extend(
        f"    {{{json.dumps(calibration_id)}, {width}u, {height}u}},"
        for calibration_id, width, height in calibrated_shapes
    )
    lines.extend([
        "  }};",
        "",
        "  constexpr bool subtitle_ocr_field_is_calibrated(",
        "    const std::uint32_t width,",
        "    const std::uint32_t height",
        "  ) {",
        "    for (const auto &shape : model_calibrated_shapes) {",
        "      if (shape.width == width && shape.height == height) {",
        "        return true;",
        "      }",
        "    }",
        "    return false;",
        "  }",
        "",
        "  struct subtitle_ocr_roi_t {",
        "    std::uint32_t top;",
        "    std::uint32_t bottom;",
        "",
        "    constexpr explicit operator bool() const { return bottom > top; }",
        "  };",
        "",
        "  struct subtitle_ocr_projected_row_t {",
        "    std::uint32_t value;",
        "    bool valid;",
        "",
        "    constexpr explicit operator bool() const { return valid; }",
        "  };",
        "",
        "  constexpr std::uint64_t subtitle_ocr_ceil_div(",
        "    const std::uint64_t numerator,",
        "    const std::uint64_t denominator",
        "  ) {",
        "    return denominator == 0u ? 0u :",
        "      numerator / denominator + (numerator % denominator != 0u);",
        "  }",
        "",
        "  constexpr subtitle_ocr_projected_row_t subtitle_ocr_project_row_ceil(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height,",
        "    const std::uint32_t detector_y",
        "  ) {",
        "    if (source_width == 0u || source_height == 0u ||",
        "        detector_y > subtitle_ocr_output_height ||",
        "        !subtitle_ocr_field_is_calibrated(field_width, field_height)) {",
        "      return {0u, false};",
        "    }",
        "    const auto requested_crop_height = subtitle_ocr_ceil_div(",
        "      static_cast<std::uint64_t>(source_width) * subtitle_ocr_crop_aspect_height,",
        "      subtitle_ocr_crop_aspect_width);",
        "    const auto crop_height = requested_crop_height < source_height ?",
        "      requested_crop_height : static_cast<std::uint64_t>(source_height);",
        "    const auto crop_top = static_cast<std::uint64_t>(source_height) - crop_height;",
        "    const auto denominator = static_cast<std::uint64_t>(source_height) *",
        "      subtitle_ocr_output_height;",
        "    const auto source_y_numerator = crop_top * subtitle_ocr_output_height +",
        "      static_cast<std::uint64_t>(detector_y) * crop_height;",
        "    const auto projected = subtitle_ocr_ceil_div(",
        "      source_y_numerator * field_height, denominator);",
        "    return {",
        "      static_cast<std::uint32_t>(projected < field_height ? projected : field_height),",
        "      true",
        "    };",
        "  }",
        "",
        "  constexpr subtitle_ocr_projected_row_t subtitle_ocr_ribbon_min_bottom(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height",
        "  ) {",
        "    return subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height,",
        "      subtitle_ocr_safe_row_bottom - subtitle_ocr_ribbon_bottom_tolerance_pixels",
        "    );",
        "  }",
        "",
        "  constexpr subtitle_ocr_roi_t subtitle_ocr_dynamic_roi(",
        "    const std::uint32_t source_width,",
        "    const std::uint32_t source_height,",
        "    const std::uint32_t field_width,",
        "    const std::uint32_t field_height",
        "  ) {",
        "    const auto top = subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height, subtitle_ocr_safe_row_top);",
        "    const auto bottom = subtitle_ocr_project_row_ceil(",
        "      source_width, source_height, field_width, field_height,",
        "      subtitle_ocr_safe_row_bottom);",
        "    return top && bottom && top.value < bottom.value ?",
        "      subtitle_ocr_roi_t {top.value, bottom.value} : subtitle_ocr_roi_t {0u, 0u};",
        "  }",
        "",
        "  constexpr bool model_calibration_supports_shape(",
        "    const model_calibration_t &calibration,",
        "    const std::uint32_t width,",
        "    const std::uint32_t height",
        "  ) {",
        "    for (const auto &shape : model_calibrated_shapes) {",
        "      if (shape.calibration_id == calibration.calibration_id &&",
        "          shape.width == width && shape.height == height) {",
        "        return true;",
        "      }",
        "    }",
        "    return false;",
        "  }",
        "",
        "  enum class constant_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in constants
    )
    lines.extend([
        f"    count = {len(constants)}u,",
        "  };",
        "",
        "  constexpr std::size_t constant_index(const constant_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
        "  inline constexpr std::size_t constant_float_count =",
        "    constant_index(constant_word_e::count);",
        "  inline constexpr std::size_t constant_vector_count = constant_float_count / 4u;",
        "  inline constexpr std::array<const char *, constant_float_count> "
        "constant_field_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in constants)
    lines.extend([
        "  }};",
        "",
        "  struct alignas(16) constants_t {",
    ])
    lines.extend(f"    float {_identifier(field['name'])};" for field in constants)
    lines.extend([
        "  };",
        "",
        "  static_assert(std::is_standard_layout_v<constants_t>);",
        "  static_assert(std::is_trivially_copyable_v<constants_t>);",
        "  static_assert(alignof(constants_t) == 16u);",
        "  static_assert(sizeof(constants_t) == constant_float_count * sizeof(float));",
        "  static_assert(sizeof(constants_t) % 16u == 0u);",
    ])
    lines.extend(
        f"  static_assert(offsetof(constants_t, {_identifier(field['name'])}) == "
        f"constant_index(constant_word_e::{_identifier(field['name'])}) * sizeof(float));"
        for field in constants
    )
    lines.extend([
        "",
        "  enum class frame_stat_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in frame_stats
    )
    lines.extend([
        f"    count = {len(frame_stats)}u,",
        "  };",
        "",
        "  constexpr std::size_t frame_stat_index(const frame_stat_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
    ])
    lines.extend(
        f"  inline constexpr std::size_t frame_stat_{_identifier(field['name'])} = "
        f"frame_stat_index(frame_stat_word_e::{_identifier(field['name'])});"
        for field in frame_stats
    )
    lines.extend([
        "",
        "  inline constexpr std::size_t frame_stats_float_count =",
        "    frame_stat_index(frame_stat_word_e::count);",
        "  inline constexpr std::size_t frame_stats_vector_count =",
        "    frame_stats_float_count / 4u;",
        "  inline constexpr std::array<const char *, frame_stats_float_count> "
        "frame_stat_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in frame_stats)
    lines.extend([
        "  }};",
        "",
        "  enum class state_gpu_encoding_e {",
        "    float_value,",
        "    uint_bits,",
        "  };",
        "",
        "  enum class state_word_e : std::size_t {",
    ])
    lines.extend(
        f"    {_identifier(field['name'])} = {field['index']}u," for field in fields
    )
    lines.extend([
        f"    count = {len(fields)}u,",
        "  };",
        "",
        "  constexpr std::size_t state_index(const state_word_e word) {",
        "    return static_cast<std::size_t>(word);",
        "  }",
        "",
    ])
    lines.extend(
        f"  inline constexpr std::size_t {_identifier(field['name'])} = "
        f"state_index(state_word_e::{_identifier(field['name'])});"
        for field in fields
    )
    lines.extend([
        "",
        "  inline constexpr std::size_t state_float_count =",
        "    state_index(state_word_e::count);",
        f"  inline constexpr std::size_t state_vector_width = {contract['vector_width']}u;",
        "  inline constexpr std::size_t state_vector_count =",
        "    state_float_count / state_vector_width;",
        "  using state_words_t = std::array<std::uint32_t, state_float_count>;",
        "",
        "  struct state_field_descriptor_t {",
        "    state_word_e word;",
        "    std::string_view name;",
        "    std::string_view json_type;",
        "    state_gpu_encoding_e gpu_encoding;",
        "    std::uint32_t initial_word;",
        "  };",
        "",
        f"  inline constexpr std::array<state_field_descriptor_t, {len(fields)}> "
        "state_fields {{",
    ])
    for field in fields:
        encoding = "float_value" if field["gpu_encoding"] == "float" else "uint_bits"
        lines.append(
            "    {state_word_e::%s, \"%s\", \"%s\", state_gpu_encoding_e::%s, %s},"
            % (
                _identifier(field["name"]),
                field["name"],
                field["type"],
                encoding,
                _initial_word_cpp(field, tag),
            )
        )
    lines.extend([
        "  }};",
        "",
        "  inline constexpr std::array<const char *, state_float_count> "
        "state_field_names {{",
    ])
    lines.extend(f'    "{field["name"]}",' for field in fields)
    lines.extend([
        "  }};",
        "",
        "  inline constexpr state_words_t state_initial_words {{",
    ])
    lines.extend(f"    {_initial_word_cpp(field, tag)}," for field in fields)
    lines.extend([
        "  }};",
        "",
        "  constexpr bool state_field_layout_matches_indices() {",
        "    for (std::size_t position = 0; position < state_fields.size(); ++position) {",
        "      if (state_index(state_fields[position].word) != position) {",
        "        return false;",
        "      }",
        "    }",
        "    return true;",
        "  }",
        "",
        "  constexpr bool state_word_is_uint_bits(const state_word_e word) {",
        "    return state_fields[state_index(word)].gpu_encoding ==",
        "      state_gpu_encoding_e::uint_bits;",
        "  }",
        "",
        "  constexpr bool contract_tag_is_finite_normal(const std::uint32_t value) {",
        "    const auto exponent = (value >> 23u) & 0xffu;",
        "    return exponent > 0u && exponent < 0xffu;",
        "  }",
        "",
        "  static_assert(frame_stats_float_count % 4u == 0u);",
        "  static_assert(state_float_count % state_vector_width == 0u);",
        "  static_assert(state_field_layout_matches_indices());",
        "  static_assert(contract_tag != 0u);",
        "  static_assert(contract_tag_is_finite_normal(contract_tag));",
        "  static_assert(state_initial_words[state_index(state_word_e::contract_tag_bits)] ==",
        "                contract_tag);",
        "}  // namespace models::depth_coordinate_v2",
        "",
    ])
    return "\n".join(lines)


def render_hlsl(contract: dict[str, Any]) -> str:
    fields = contract["shadow_state"]["fields"]
    constant_buffer = contract["constant_buffer"]
    constants = constant_buffer["fields"]
    frame_stats = contract["frame_stats"]["fields"]
    defaults = contract["calibrated_defaults"]
    subtitle_ocr = contract["subtitle_ocr"]
    ocr_input = subtitle_ocr["input_tensor"]
    ocr_output = subtitle_ocr["output_tensor"]
    field_policy = subtitle_ocr["field_policy"]
    ocr_record = subtitle_ocr["ocr_record"]
    locator_state = subtitle_ocr["locator_state"]
    condition_params = subtitle_ocr["condition_params"]
    calibrated_shapes = [
        (shape["width"], shape["height"])
        for calibration in contract["model_calibrations"]
        for shape in calibration["calibrated_input_shapes"]
    ]
    calibrated_max_dimension = max(
        max(width, height) for width, height in calibrated_shapes
    )
    limiter_container_q_limit = math.ceil(
        _float32(defaults["direct_container_limit"]) * LIMITER_Q_SCALE)
    limiter_horizontal_step_q_numerator = int(
        float(defaults["max_horizontal_slope"]) * LIMITER_Q_SCALE)
    limiter_vertical_step_q_numerator = int(
        float(defaults["max_vertical_shear"]) * LIMITER_Q_SCALE)
    shape_macros = [
        macro
        for index, (width, height) in enumerate(calibrated_shapes)
        for macro in (
            f"#define V2_MODEL_CALIBRATED_SHAPE_WIDTH_{index} {width}u",
            f"#define V2_MODEL_CALIBRATED_SHAPE_HEIGHT_{index} {height}u",
        )
    ]
    shape_predicate = " ||\n           ".join(
        f"(field_width == V2_MODEL_CALIBRATED_SHAPE_WIDTH_{index} && "
        f"field_height == V2_MODEL_CALIBRATED_SHAPE_HEIGHT_{index})"
        for index in range(len(calibrated_shapes))
    )
    tag = contract_tag(contract)
    components = ("x", "y", "z", "w")
    lines = [
        "// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.",
        "// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.",
        "#ifndef DEPTH_COORDINATE_V2_CONTRACT_GENERATED_HLSL",
        "#define DEPTH_COORDINATE_V2_CONTRACT_GENERATED_HLSL",
        "",
        f"#define V2_CONTRACT_SCHEMA {contract['schema']}u",
        f"#define V2_CONTRACT_TAG 0x{tag:08X}u",
        f"#define V2_SHADOW_STATE_WORD_COUNT {len(fields)}u",
        f"#define V2_SHADOW_STATE_VECTOR_COUNT {len(fields) // contract['vector_width']}u",
        f"#define V2_SUBTITLE_OCR_CONTRACT_SCHEMA {subtitle_ocr['schema']}u",
        f"#define V2_OCR_INPUT_N {ocr_input['shape'][0]}u",
        f"#define V2_OCR_INPUT_C {ocr_input['shape'][1]}u",
        f"#define V2_OCR_INPUT_HEIGHT {ocr_input['shape'][2]}u",
        f"#define V2_OCR_INPUT_WIDTH {ocr_input['shape'][3]}u",
        f"#define V2_OCR_OUTPUT_N {ocr_output['shape'][0]}u",
        f"#define V2_OCR_OUTPUT_C {ocr_output['shape'][1]}u",
        f"#define V2_OCR_OUTPUT_HEIGHT {ocr_output['shape'][2]}u",
        f"#define V2_OCR_OUTPUT_WIDTH {ocr_output['shape'][3]}u",
        f"#define V2_OCR_IMAGENET_MEAN_B {_float_literal(ocr_input['imagenet_mean'][0])}",
        f"#define V2_OCR_IMAGENET_MEAN_G {_float_literal(ocr_input['imagenet_mean'][1])}",
        f"#define V2_OCR_IMAGENET_MEAN_R {_float_literal(ocr_input['imagenet_mean'][2])}",
        f"#define V2_OCR_IMAGENET_STD_B {_float_literal(ocr_input['imagenet_std'][0])}",
        f"#define V2_OCR_IMAGENET_STD_G {_float_literal(ocr_input['imagenet_std'][1])}",
        f"#define V2_OCR_IMAGENET_STD_R {_float_literal(ocr_input['imagenet_std'][2])}",
        f"#define V2_OCR_RECORD_SCHEMA {ocr_record['schema']}u",
        f"#define V2_OCR_RECORD_TAG 0x{ocr_record['tag']:08X}u",
        f"#define V2_OCR_RECORD_WORD_COUNT {ocr_record['word_count']}u",
        f"#define V2_OCR_RECORD_HEADER_WORD_COUNT {ocr_record['header_word_count']}u",
        f"#define V2_OCR_BOX_WORD_COUNT {ocr_record['box_word_count']}u",
        f"#define V2_OCR_BOX_FLAG_RIBBON {ocr_record['box_flag_ribbon']}u",
        f"#define V2_OCR_BOX_KNOWN_FLAGS {ocr_record['box_known_flags']}u",
        f"#define V2_OCR_RAW_BOX_OFFSET {ocr_record['raw_box_offset']}u",
        f"#define V2_OCR_RAW_BOX_CAPACITY {ocr_record['raw_box_capacity']}u",
        f"#define V2_OCR_FINAL_BOX_OFFSET {ocr_record['final_box_offset']}u",
        f"#define V2_OCR_FINAL_BOX_CAPACITY {ocr_record['final_box_capacity']}u",
        f"#define V2_MODEL_CALIBRATED_SHAPE_COUNT {len(calibrated_shapes)}u",
        f"#define V2_MODEL_CALIBRATED_MAX_DIMENSION {calibrated_max_dimension}u",
        f"#define V2_LIMITER_GROUP_THREADS {LIMITER_GROUP_THREADS}u",
        f"#define V2_LIMITER_Q_FRACTION_BITS {LIMITER_Q_FRACTION_BITS}u",
        f"#define V2_LIMITER_Q_SCALE {LIMITER_Q_SCALE}.0f",
        f"#define V2_LIMITER_CONTAINER_Q_LIMIT {limiter_container_q_limit}",
        f"#define V2_LIMITER_HORIZONTAL_STEP_Q_NUMERATOR "
        f"{limiter_horizontal_step_q_numerator}u",
        f"#define V2_LIMITER_VERTICAL_STEP_Q_NUMERATOR "
        f"{limiter_vertical_step_q_numerator}u",
        *shape_macros,
        f"#define V2_OCR_SAFE_ROW_TOP {field_policy['ocr_safe_row_top']}u",
        f"#define V2_OCR_SAFE_ROW_BOTTOM {field_policy['ocr_safe_row_bottom']}u",
        f"#define V2_OCR_ACTIVE_PROBABILITY_THRESHOLD "
        f"{_float_literal(field_policy['detector_active_probability_threshold'])}",
        f"#define V2_OCR_MIN_MEAN_SCORE "
        f"{_float_literal(field_policy['detector_min_mean_score'])}",
        f"#define V2_OCR_CROP_ASPECT_WIDTH "
        f"{field_policy['source_crop_aspect_width']}u",
        f"#define V2_OCR_CROP_ASPECT_HEIGHT "
        f"{field_policy['source_crop_aspect_height']}u",
        f"#define V2_OCR_TEXT_JOIN_GAP_CELLS "
        f"{field_policy['text_join_gap_cells']}u",
        f"#define V2_OCR_RIBBON_JOIN_GAP_CELLS "
        f"{field_policy['ribbon_join_gap_cells']}u",
        f"#define V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS "
        f"{field_policy['ribbon_structural_gap_min_cells']}u",
        f"#define V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS "
        f"{field_policy['ribbon_min_structural_gaps']}u",
        f"#define V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR "
        f"{field_policy['ribbon_min_width_numerator']}u",
        f"#define V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR "
        f"{field_policy['ribbon_min_width_denominator']}u",
        f"#define V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS "
        f"{field_policy['ribbon_bottom_tolerance_pixels']}u",
        f"#define V2_OCR_RIBBON_COVER_PAD_LIMIT "
        f"{field_policy['ribbon_cover_pad_limit']}u",
        f"#define V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR "
        f"{field_policy['locator_max_width_numerator']}u",
        f"#define V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR "
        f"{field_policy['locator_max_width_denominator']}u",
        f"#define V2_SUBTITLE_LOCATOR_MIN_WIDTH_CELLS "
        f"{field_policy['locator_min_width_cells']}u",
        f"#define V2_SUBTITLE_LOCATOR_MIN_HEIGHT_CELLS "
        f"{field_policy['locator_min_height_cells']}u",
        f"#define V2_SUBTITLE_LOCATOR_MIN_ASPECT_NUMERATOR "
        f"{field_policy['locator_min_aspect_numerator']}u",
        f"#define V2_SUBTITLE_LOCATOR_MIN_ASPECT_DENOMINATOR "
        f"{field_policy['locator_min_aspect_denominator']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_NUMERATOR "
        f"{field_policy['locator_provisional_min_vertical_overlap_numerator']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_DENOMINATOR "
        f"{field_policy['locator_provisional_min_vertical_overlap_denominator']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_HEIGHT_RATIO "
        f"{field_policy['locator_provisional_max_height_ratio']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_CENTER_Y_DELTA_SHORTER_HEIGHT "
        f"{field_policy['locator_provisional_max_center_y_delta_shorter_height']}u",
        f"#define V2_SUBTITLE_LOCATOR_CORNER_EDGE_DIVISOR "
        f"{field_policy['locator_corner_edge_divisor']}u",
        f"#define V2_SUBTITLE_LOCATOR_CORNER_BOTTOM_ROWS "
        f"{field_policy['locator_corner_bottom_rows']}u",
        f"#define V2_SUBTITLE_LOCATOR_MATCH_IOU_THRESHOLD "
        f"{_float_literal(field_policy['locator_match_iou_threshold'])}",
        f"#define V2_SUBTITLE_LOCATOR_DEATH_GRACE_OBSERVATIONS "
        f"{field_policy['locator_death_grace_observations']}u",
        f"#define V2_SUBTITLE_TARGET_HORIZONTAL_FALLBACK_MAX_RADIUS_STEPS "
        f"{field_policy['locator_target_horizontal_fallback_max_radius_steps']}u",
        f"#define V2_SUBTITLE_TARGET_HORIZONTAL_STEP_DENOMINATOR "
        f"{field_policy['locator_target_horizontal_step_denominator']}u",
        f"#define V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS "
        f"{_float_literal(field_policy['locator_target_max_row_iqr_binocular_source_pixels'])}",
        f"#define V2_SUBTITLE_TARGET_MAX_ROW_MEDIAN_DELTA_BINOCULAR_SOURCE_PIXELS "
        f"{_float_literal(field_policy['locator_target_max_row_median_delta_binocular_source_pixels'])}",
        f"#define V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS "
        f"{_float_literal(field_policy['locator_target_max_residual_binocular_source_pixels'])}",
        f"#define V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS "
        f"{field_policy['locator_target_max_unreliable_holds']}u",
        f"#define V2_SUBTITLE_TARGET_DEADBAND_BINOCULAR_SOURCE_PIXELS "
        f"{_float_literal(field_policy['locator_target_deadband_binocular_source_pixels'])}",
        f"#define V2_SUBTITLE_TARGET_EMA_ALPHA "
        f"{_float_literal(field_policy['locator_target_ema_alpha'])}",
        f"#define V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS "
        f"{_float_literal(field_policy['locator_target_max_slew_binocular_source_pixels'])}",
        f"#define V2_SUBTITLE_LOCATOR_STATE_SCHEMA {locator_state['schema']}u",
        f"#define V2_SUBTITLE_LOCATOR_STATE_TAG 0x{locator_state['tag']:08X}u",
        f"#define V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT {locator_state['word_count']}u",
        f"#define V2_SUBTITLE_LOCATOR_HEADER_WORD_COUNT {locator_state['header_word_count']}u",
        f"#define V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY {locator_state['rectangle_capacity']}u",
        f"#define V2_SUBTITLE_LOCATOR_OWNER_OFFSET {locator_state['owner_offset']}u",
        f"#define V2_SUBTITLE_LOCATOR_PENDING_OFFSET {locator_state['pending_offset']}u",
        f"#define V2_SUBTITLE_LOCATOR_CURRENT_OFFSET {locator_state['current_offset']}u",
        f"#define V2_SUBTITLE_LOCATOR_KIND_WORD {locator_state['kind_word']}u",
        f"#define V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT {locator_state['owner_kind_shift']}u",
        f"#define V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT {locator_state['pending_kind_shift']}u",
        f"#define V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT {locator_state['current_kind_shift']}u",
        f"#define V2_SUBTITLE_LOCATOR_KIND_MASK {locator_state['kind_mask']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_CURRENT_FLAG "
        f"{locator_state['provisional_current_flag']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD "
        f"{locator_state['provisional_target_word']}u",
        "#define V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD "
        f"{locator_state['provisional_fade_word']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_SCHEMA {condition_params['schema']}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_TAG 0x{condition_params['tag']:08X}u",
        f"#define V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT {condition_params['word_count']}u",
        "#define V2_OCR_SHADER_CONTRACT_GENERATED 1u",
        f"#define V2_DIRECT_CONTAINER_LIMIT "
        f"{_float_literal(defaults['direct_container_limit'])}",
        f"#define V2_MAX_VERTICAL_SHEAR "
        f"{_float_literal(defaults['max_vertical_shear'])}",
        f"#define V2_VERTICAL_MAJORANT_SHARE "
        f"{_float_literal(defaults['vertical_majorant_share'])}",
        # The live pixel shader authenticates the published convergence curve against this
        # compile-time contract value. It must never read the producer constant buffer, which is
        # not bound at the warp draw's pixel stage.
        f"#define V2_CONVERGENCE_CURVE_DEFAULT "
        f"{_float_literal(defaults['convergence_curve_default'])}",
        "static const float v2_max_vertical_shear = V2_MAX_VERTICAL_SHEAR;",
        "static const float v2_vertical_majorant_share = V2_VERTICAL_MAJORANT_SHARE;",
        "",
        "bool V2SubtitleOcrFieldIsCalibrated(uint field_width, uint field_height) {",
        f"    return {shape_predicate};",
        "}",
        "",
        "// Resolve the authenticated bottom crop without uint64 arithmetic.",
        "bool V2SubtitleOcrComputeCrop(",
        "    uint source_width, uint source_height, out uint crop_top, out uint crop_height) {",
        "    crop_top = 0u;",
        "    crop_height = 0u;",
        "    if (source_width == 0u || source_height == 0u) return false;",
        "    uint crop_quotient = source_width / V2_OCR_CROP_ASPECT_WIDTH;",
        "    uint crop_remainder = source_width % V2_OCR_CROP_ASPECT_WIDTH;",
        "    if (crop_quotient > 0xffffffffu / V2_OCR_CROP_ASPECT_HEIGHT ||",
        "        crop_remainder > 0xffffffffu / V2_OCR_CROP_ASPECT_HEIGHT) return false;",
        "    crop_height = min(",
        "        source_height,",
        "        crop_quotient * V2_OCR_CROP_ASPECT_HEIGHT +",
        "        (crop_remainder * V2_OCR_CROP_ASPECT_HEIGHT +",
        "         V2_OCR_CROP_ASPECT_WIDTH - 1u) / V2_OCR_CROP_ASPECT_WIDTH);",
        "    crop_top = source_height - crop_height;",
        "    return crop_height != 0u;",
        "}",
        "",
        "// Exact ceil projection of a detector row edge through that crop into the real",
        "// tensor-content rectangle.",
        "bool V2SubtitleOcrProjectContentRowCeil(",
        "    uint source_width, uint source_height, uint field_width, uint field_height,",
        "    uint4 tensor_content, uint detector_y, out uint projected_y) {",
        "    projected_y = 0u;",
        "    if (source_width == 0u || source_height == 0u || field_height == 0u ||",
        "        detector_y > V2_OCR_OUTPUT_HEIGHT ||",
        "        !V2SubtitleOcrFieldIsCalibrated(field_width, field_height) ||",
        "        tensor_content.x >= tensor_content.z || tensor_content.y >= tensor_content.w ||",
        "        tensor_content.z > field_width || tensor_content.w > field_height) return false;",
        "    uint crop_top;",
        "    uint crop_height;",
        "    if (!V2SubtitleOcrComputeCrop(",
        "            source_width, source_height, crop_top, crop_height)) return false;",
        "    if (source_height > 0xffffffffu / V2_OCR_OUTPUT_HEIGHT) return false;",
        "    uint denominator = V2_OCR_OUTPUT_HEIGHT * source_height;",
        "    uint content_height = tensor_content.w - tensor_content.y;",
        "    if (denominator == 0u || denominator > 0xffffffffu / content_height) return false;",
        "    uint source_y_numerator =",
        "        crop_top * V2_OCR_OUTPUT_HEIGHT + detector_y * crop_height;",
        "    uint scaled = source_y_numerator * content_height;",
        "    projected_y = tensor_content.y + min(",
        "        scaled / denominator + (scaled % denominator != 0u ? 1u : 0u),",
        "        content_height);",
        "    return true;",
        "}",
        "",
        f"cbuffer {constant_buffer['name']} : register({constant_buffer['register']}) {{",
    ]
    lines.extend(f"    float v2_{field['name']};" for field in constants)
    lines.extend([
        "};",
        "",
        f"#define V2_CONSTANT_WORD_COUNT {len(constants)}u",
        f"#define V2_CONSTANT_VECTOR_COUNT {len(constants) // contract['vector_width']}u",
        f"#define V2_FRAME_STATS_WORD_COUNT {len(frame_stats)}u",
        f"#define V2_FRAME_STATS_VECTOR_COUNT "
        f"{len(frame_stats) // contract['vector_width']}u",
        "",
    ])
    for field in frame_stats:
        upper = _upper_identifier(field["name"])
        vector = int(field["index"]) // contract["vector_width"]
        component = components[int(field["index"]) % contract["vector_width"]]
        lines.extend([
            f"#define V2_FRAME_STATS_WORD_{upper} {field['index']}u",
            f"#define V2_FRAME_STATS_VECTOR_{upper} {vector}u",
            f"#define V2_FRAME_STATS_{upper}(value) ((value).{component})",
        ])
    lines.extend([
        "",
    ])
    for field in fields:
        upper = _upper_identifier(field["name"])
        vector = int(field["index"]) // contract["vector_width"]
        component = components[int(field["index"]) % contract["vector_width"]]
        lines.extend([
            f"#define V2_STATE_WORD_{upper} {field['index']}u",
            f"#define V2_STATE_VECTOR_{upper} {vector}u",
            f"#define V2_STATE_{upper}(value) ((value).{component})",
        ])
    lines.extend(["", "#endif", ""])
    return "\n".join(lines)


def render_hlsl_ocr_assertions() -> str:
    """Render the shared compile-time OCR8/SLR13 consumer invariants."""

    return "\n".join([
        "// Generated by tools/sbsbench/generate_depth_coordinate_v2_contract.py.",
        "// Edit tools/sbsbench/contracts/depth-coordinate-v2-v1.json instead.",
        "#ifndef DEPTH_COORDINATE_V2_OCR_ASSERT_GENERATED_HLSL",
        "#define DEPTH_COORDINATE_V2_OCR_ASSERT_GENERATED_HLSL",
        "",
        '#include "include/depth_coordinate_v2_contract.generated.hlsl"',
        "",
        "#if !defined(V2_OCR_SHADER_CONTRACT_GENERATED)",
        '#error "Complete generated V2 OCR8/SLR13 contract is required"',
        "#endif",
        "",
        "#if !defined(V2_SUBTITLE_TARGET_MAX_ROW_IQR_BINOCULAR_SOURCE_PIXELS) || \\",
        "    !defined(V2_SUBTITLE_TARGET_MAX_ROW_MEDIAN_DELTA_BINOCULAR_SOURCE_PIXELS) || \\",
        "    !defined(V2_SUBTITLE_TARGET_MAX_RESIDUAL_BINOCULAR_SOURCE_PIXELS) || \\",
        "    !defined(V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_CORNER_EDGE_DIVISOR) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_CORNER_BOTTOM_ROWS) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_NUMERATOR) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_DENOMINATOR) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_HEIGHT_RATIO) || \\",
        "    !defined(V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_CENTER_Y_DELTA_SHORTER_HEIGHT) || \\",
        "    !defined(V2_SUBTITLE_TARGET_DEADBAND_BINOCULAR_SOURCE_PIXELS) || \\",
        "    !defined(V2_SUBTITLE_TARGET_EMA_ALPHA) || \\",
        "    !defined(V2_SUBTITLE_TARGET_MAX_SLEW_BINOCULAR_SOURCE_PIXELS)",
        '#error "Complete generated V2 subtitle target policy is required"',
        "#endif",
        "",
        "#if V2_MODEL_CALIBRATED_SHAPE_COUNT != 6 || \\",
        "    V2_OCR_SAFE_ROW_TOP >= V2_OCR_SAFE_ROW_BOTTOM || \\",
        "    V2_OCR_SAFE_ROW_BOTTOM > V2_OCR_OUTPUT_HEIGHT || \\",
        "    V2_OCR_CROP_ASPECT_WIDTH == 0u || V2_OCR_CROP_ASPECT_HEIGHT == 0u || \\",
        "    V2_OCR_CROP_ASPECT_HEIGHT > V2_OCR_CROP_ASPECT_WIDTH || \\",
        "    V2_OCR_BOX_FLAG_RIBBON != 1u || V2_OCR_BOX_KNOWN_FLAGS != 1u || \\",
        "    V2_OCR_TEXT_JOIN_GAP_CELLS == 0u || \\",
        "    V2_OCR_TEXT_JOIN_GAP_CELLS >= V2_OCR_RIBBON_JOIN_GAP_CELLS || \\",
        "    V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS == 0u || \\",
        "    V2_OCR_RIBBON_JOIN_GAP_CELLS < V2_OCR_RIBBON_STRUCTURAL_GAP_MIN_CELLS || \\",
        "    V2_OCR_RIBBON_MIN_STRUCTURAL_GAPS == 0u || \\",
        "    V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR == 0u || \\",
        "    V2_OCR_RIBBON_MIN_WIDTH_NUMERATOR >= V2_OCR_RIBBON_MIN_WIDTH_DENOMINATOR || \\",
        "    V2_OCR_RIBBON_BOTTOM_TOLERANCE_PIXELS >= V2_OCR_SAFE_ROW_BOTTOM || \\",
        "    V2_OCR_RIBBON_COVER_PAD_LIMIT == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_MAX_WIDTH_NUMERATOR >= \\",
        "        V2_SUBTITLE_LOCATOR_MAX_WIDTH_DENOMINATOR || \\",
        "    V2_SUBTITLE_LOCATOR_MIN_WIDTH_CELLS == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_MIN_HEIGHT_CELLS == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_MIN_ASPECT_NUMERATOR == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_MIN_ASPECT_DENOMINATOR == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_CORNER_EDGE_DIVISOR <= 1u || \\",
        "    V2_SUBTITLE_LOCATOR_CORNER_BOTTOM_ROWS == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_NUMERATOR == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_DENOMINATOR < \\",
        "        V2_SUBTITLE_LOCATOR_PROVISIONAL_MIN_VERTICAL_OVERLAP_NUMERATOR || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_HEIGHT_RATIO == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_MAX_CENTER_Y_DELTA_SHORTER_HEIGHT == 0u || \\",
        "    V2_SUBTITLE_LOCATOR_DEATH_GRACE_OBSERVATIONS == 0u || \\",
        "    V2_SUBTITLE_TARGET_MAX_UNRELIABLE_HOLDS == 0u || \\",
        "    V2_OCR_RECORD_HEADER_WORD_COUNT != V2_OCR_RAW_BOX_OFFSET || \\",
        "    V2_OCR_FINAL_BOX_OFFSET != V2_OCR_RAW_BOX_OFFSET + \\",
        "        V2_OCR_RAW_BOX_CAPACITY * V2_OCR_BOX_WORD_COUNT || \\",
        "    V2_OCR_RECORD_WORD_COUNT != V2_OCR_FINAL_BOX_OFFSET + \\",
        "        V2_OCR_FINAL_BOX_CAPACITY * V2_OCR_BOX_WORD_COUNT || \\",
        "    V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY != 4u || \\",
        "    V2_SUBTITLE_LOCATOR_HEADER_WORD_COUNT != V2_SUBTITLE_LOCATOR_OWNER_OFFSET || \\",
        "    V2_SUBTITLE_LOCATOR_PENDING_OFFSET != V2_SUBTITLE_LOCATOR_OWNER_OFFSET + \\",
        "        V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY * 4u || \\",
        "    V2_SUBTITLE_LOCATOR_CURRENT_OFFSET != V2_SUBTITLE_LOCATOR_PENDING_OFFSET + \\",
        "        V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY * 4u || \\",
        "    V2_SUBTITLE_LOCATOR_STATE_WORD_COUNT != V2_SUBTITLE_LOCATOR_CURRENT_OFFSET + \\",
        "        V2_SUBTITLE_LOCATOR_RECTANGLE_CAPACITY * 4u || \\",
        "    V2_SUBTITLE_LOCATOR_KIND_WORD != 31u || \\",
        "    V2_SUBTITLE_LOCATOR_OWNER_KIND_SHIFT != 0u || \\",
        "    V2_SUBTITLE_LOCATOR_PENDING_KIND_SHIFT != 4u || \\",
        "    V2_SUBTITLE_LOCATOR_CURRENT_KIND_SHIFT != 8u || \\",
        "    V2_SUBTITLE_LOCATOR_KIND_MASK != 15u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_CURRENT_FLAG != 16u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_TARGET_WORD != 29u || \\",
        "    V2_SUBTITLE_LOCATOR_PROVISIONAL_FADE_WORD != 30u || \\",
        "    V2_SUBTITLE_TARGET_HORIZONTAL_FALLBACK_MAX_RADIUS_STEPS != 2u || \\",
        "    V2_SUBTITLE_TARGET_HORIZONTAL_STEP_DENOMINATOR != 16u || \\",
        "    V2_SUBTITLE_CONDITION_PARAM_WORD_COUNT != 6u",
        '#error "Generated V2 OCR8/SLR13 contract invariants are inconsistent"',
        "#endif",
        "",
        "#endif",
        "",
    ])


def _write_or_check(path: Path, expected: str, check: bool) -> bool:
    if check:
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError:
            print(f"missing generated contract: {path}", file=sys.stderr)
            return False
        if actual != expected:
            print(f"stale generated contract: {path}", file=sys.stderr)
            return False
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if committed generated files differ from the manifest",
    )
    parser.add_argument(
        "--refresh-shader-identity",
        action="store_true",
        help=("regenerate the non-circular HLSL tag, refresh the independent complete "
              "shader closure digest in the manifest, then regenerate both language contracts"),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check and args.refresh_shader_identity:
        parser.error("--check and --refresh-shader-identity are mutually exclusive")
    if args.refresh_shader_identity:
        contract = load_contract(
            args.manifest,
            verify_shader_source_closure=False,
            verify_preprocess_source_closure=False,
        )
        preprocess_digest = shader_source_closure_sha256()
        for calibration in contract["model_calibrations"]:
            calibration["preprocess"]["source_closure_sha256"] = preprocess_digest
        # The GPU tag deliberately excludes only the shader-body digest, so this first HLSL
        # generation is final. Hash the complete preprocessing, overlay, cut/history, and
        # coordinate-producer closure, record that independent
        # identity, and render again; the second HLSL must be byte-identical by construction.
        first_hlsl = render_hlsl(contract)
        first_ocr_assertions = render_hlsl_ocr_assertions()
        _write_or_check(HLSL_TARGET, first_hlsl, False)
        _write_or_check(HLSL_OCR_ASSERT_TARGET, first_ocr_assertions, False)
        contract["shader_implementation"]["source_closure_sha256"] = (
            shader_source_closure_sha256(specs=PARALLAX_V2_SHADER_SPECS))
        contract = validate_contract(contract)
        second_hlsl = render_hlsl(contract)
        second_ocr_assertions = render_hlsl_ocr_assertions()
        if (second_hlsl != first_hlsl or
                second_ocr_assertions != first_ocr_assertions):
            raise RuntimeError("shader identity refresh is not idempotent")
        args.manifest.write_text(
            json.dumps(contract, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        contract = load_contract(args.manifest)
    valid = _write_or_check(CPP_TARGET, render_cpp(contract), args.check)
    valid &= _write_or_check(HLSL_TARGET, render_hlsl(contract), args.check)
    valid &= _write_or_check(
        HLSL_OCR_ASSERT_TARGET, render_hlsl_ocr_assertions(), args.check)
    try:
        validate_renderer_source_closure_pins()
    except ValueError as exc:
        print(f"stale: {exc}", file=sys.stderr)
        valid = False
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
