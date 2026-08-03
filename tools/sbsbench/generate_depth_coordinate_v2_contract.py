#!/usr/bin/env python3
"""Generate native and HLSL declarations for the complete depth-coordinate-v2 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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

EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "vector_width", "calibrated_defaults", "constant_buffer", "frame_stats",
    "shadow_state", "model_calibrations", "capture_provenance", "shader_implementation",
}
EXPECTED_FIELD_KEYS = {"index", "name", "type", "gpu_encoding", "initial"}
EXPECTED_LAYOUT_FIELD_KEYS = {"index", "name", "type"}
EXPECTED_DEFAULT_NAMES = (
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "near_tail_probe_u",
    "near_tail_coverage_low",
    "near_tail_coverage_high",
    "near_log_tau_dense",
    "gain_per_pop",
    "reference_pop_strength",
    "direct_container_limit",
    "max_horizontal_slope",
    "max_vertical_shear",
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
EXPECTED_CONSTANT_FIELD_NAMES = (
    "raw_coordinate_scale",
    "collapse_abs_epsilon",
    "far_tau",
    "near_log_tau",
    "requested_gain",
    "max_horizontal_slope",
    "direct_container_limit",
    "convergence_curve_default",
    "near_tail_probe_u",
    "near_tail_coverage_low",
    "near_tail_coverage_high",
    "near_log_tau_dense",
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
    "latched_near_tail_coverage",
    "effective_near_log_tau",
    "latched_near_tail_count",
    "near_shoulder_reserved",
)
CONTRACT_TAG_SENTINEL = "contract_tag"
PREPROCESS_SHADER_ROOT = (
    ROOT / "src_assets" / "windows" / "assets" / "shaders" / "directx")
PREPROCESS_SHADER_SPECS = (("rgb_to_nchw_cs.hlsl", "main", "cs_5_0"),)
PARALLAX_V2_SHADER_SPECS = (
    ("depth_coordinate_v2_moments_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_frame_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_near_coverage_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_state_resolve_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_map_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_vertical_limit_cs.hlsl", "main", "cs_5_0"),
    ("depth_coordinate_v2_limit_cs.hlsl", "main", "cs_5_0"),
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
        contract: Any, *, verify_shader_source_closure: bool = True) -> dict[str, Any]:
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
    if calibrated_defaults["near_tail_probe_u"] < 1.0:
        raise ValueError("near_tail_probe_u must be at least the near-curve knee")
    coverage_low = calibrated_defaults["near_tail_coverage_low"]
    coverage_high = calibrated_defaults["near_tail_coverage_high"]
    if not 0.0 <= coverage_low < coverage_high <= 1.0:
        raise ValueError("near-tail coverage thresholds must satisfy 0 <= low < high <= 1")
    if calibrated_defaults["near_log_tau_dense"] >= calibrated_defaults["near_log_tau"]:
        raise ValueError("near_log_tau_dense must be smaller than near_log_tau")

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
        if source_closure_sha256 != shader_source_closure_sha256():
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
        path: Path = MANIFEST, *, verify_shader_source_closure: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return validate_contract(
            json.load(stream),
            verify_shader_source_closure=verify_shader_source_closure,
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
        "  inline constexpr std::string_view direct_parallax_decode_expression = "
        f'"(encoded * 2 - 1) * {json.dumps(float(defaults["direct_container_limit"]))}";',
        "  static_assert(convergence_curve_default == 0.0f);",
        "  static_assert(max_horizontal_slope > 0.0f && max_horizontal_slope < 1.0f);",
        "  static_assert(max_vertical_shear > 0.0f);",
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
        f"#define V2_DIRECT_CONTAINER_LIMIT "
        f"{_float_literal(defaults['direct_container_limit'])}",
        f"#define V2_MAX_VERTICAL_SHEAR "
        f"{_float_literal(defaults['max_vertical_shear'])}",
        "static const float v2_max_vertical_shear = V2_MAX_VERTICAL_SHEAR;",
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
        help=("regenerate the non-circular HLSL tag, refresh the independent seven-shader "
              "closure digest in the manifest, then regenerate both language contracts"),
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check and args.refresh_shader_identity:
        parser.error("--check and --refresh-shader-identity are mutually exclusive")
    if args.refresh_shader_identity:
        contract = load_contract(
            args.manifest,
            verify_shader_source_closure=False,
        )
        # The GPU tag deliberately excludes only the shader-body digest, so this first HLSL
        # generation is final.  Hash its exact seven-shader closure, record that independent
        # identity, and render again; the second HLSL must be byte-identical by construction.
        first_hlsl = render_hlsl(contract)
        _write_or_check(HLSL_TARGET, first_hlsl, False)
        contract["shader_implementation"]["source_closure_sha256"] = (
            shader_source_closure_sha256(specs=PARALLAX_V2_SHADER_SPECS))
        contract = validate_contract(contract)
        second_hlsl = render_hlsl(contract)
        if second_hlsl != first_hlsl:
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
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
